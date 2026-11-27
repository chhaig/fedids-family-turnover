"""Amended copy of cross_testbed_runner_v1.py (rules_arch_runner_v2): adds coordinate-wise median and Krum (f=1)
and an optional second architecture (--arch B, hidden widths 256-128-64). Everything else is unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def aggregate(ds, method, weights, root):
    if method == 'root_only':
        return root
    if method == 'fedavg':
        out = torch.zeros_like(ds[0])
        for d, w in zip(ds, weights):
            out.add_(d, alpha=w)
        return out
    if method == 'trimmed_mean_f1':
        return ds.sort(dim=0).values[1:-1].mean(dim=0)
    if method == 'coord_median':
        v = ds.sort(dim=0).values; n = v.shape[0]
        return v[n // 2] if n % 2 else v[n // 2 - 1:n // 2 + 1].mean(dim=0)
    if method == 'krum_f1':
        d2 = torch.cdist(ds.double(), ds.double()) ** 2; n = ds.shape[0]; k = n - 1 - 2
        score = d2.sort(dim=1).values[:, 1:k + 1].sum(dim=1)
        return ds[int(torch.argmin(score))]
    assert method in ['fltrust', 'fltrust_server']
    d, r = ds.double(), root.double()
    rn = torch.linalg.vector_norm(r)
    norms = torch.linalg.vector_norm(d, dim=1)
    unit = d / norms.clamp_min(1e-30)[:, None]
    trust = (unit @ (r / rn.clamp_min(1e-30))).clamp(0, 1)
    if method == 'fltrust_server':
        return ((torch.sum(unit * (trust * rn)[:, None], 0) + r) / (trust.sum() + 1)).to(ds.dtype)
    return (torch.sum(unit * (trust * rn)[:, None], 0) / trust.sum().clamp_min(1e-30)).to(ds.dtype)


def trust_values(ds, root):
    d, r = ds.double(), root.double()
    u = d / torch.linalg.vector_norm(d, dim=1).clamp_min(1e-30)[:, None]
    return (u @ (r / torch.linalg.vector_norm(r).clamp_min(1e-30))).clamp(0, 1)


def adaptive(ds, method, weights, root, attacker, config):
    """Exact declared one-round surrogate; no data or evaluator access."""
    ds, root = ds.detach(), root.detach()
    clean = aggregate(ds, method, weights, root).detach()
    direction = clean.sign()
    radius = (torch.linalg.vector_norm(ds.double(), dim=1).median() * config['radius_multiplier']).to(ds.dtype)
    def project(v):
        return v * torch.clamp(radius / torch.linalg.vector_norm(v).clamp_min(1e-30), max=1)
    def replace(v):
        return torch.cat((ds[:attacker], v[None], ds[attacker + 1:]), dim=0)
    def objective(v):
        return torch.mean(direction * (clean - aggregate(replace(v), method, weights, root)))
    starts = [project(ds[attacker]), project(-clean * (radius / torch.linalg.vector_norm(clean).clamp_min(1e-30))),
              project(-direction * (radius / torch.linalg.vector_norm(direction).clamp_min(1e-30)))]
    if method.startswith('fltrust'):
        starts.append(project(root * (radius / torch.linalg.vector_norm(root).clamp_min(1e-30))))
    values = [float(objective(v)) for v in starts]
    k = int(np.argmax(values))
    best, best_value = starts[k].detach().clone(), values[k]
    current = best.clone()
    trace = [best_value]
    gradient_steps = 0
    for _ in range(config['steps']):
        current = current.detach().requires_grad_(True)
        value = objective(current)
        grad, = torch.autograd.grad(value, current)
        assert torch.isfinite(grad).all()
        norm = torch.linalg.vector_norm(grad)
        if float(norm) > 0:
            current = project(current.detach() + config['step_radius_fraction'] * radius * grad.detach() / norm)
            gradient_steps += 1
        else:
            current = current.detach()
        score = float(objective(current))
        trace.append(score)
        if score > best_value:
            best, best_value = current.detach().clone(), score
    submitted_norm = float(torch.linalg.vector_norm(best.double()))
    assert submitted_norm <= float(radius) * (1 + 2e-6) + 1e-12
    assert best_value >= max(values) - 1e-12
    info = dict(radius=float(radius), submitted_norm=submitted_norm, initial_objectives=values,
                final_objective=best_value, initial_best=max(values), objective_trace=trace,
                gradient_steps=gradient_steps, selected_initial=k,
                knowledge='current clean updates and root update; no evaluation data')
    return best, info


def fixtures(device='cpu'):
    d = torch.tensor([[0, 0, 1000], [1, 5, 1], [2, 4, 2], [3, 3, 3], [4, 2, 4], [1000, 1, 0]], dtype=torch.float64, device=device)
    assert torch.equal(aggregate(d, 'trimmed_mean_f1', [1 / 6] * 6, d[0]), torch.full((3,), 2.5, dtype=d.dtype, device=device))
    ds = torch.tensor([[1., 0], [0, 1], [-1, 0], [1, 1], [0, 0], [2, 0]], dtype=torch.float64, device=device)
    root = ds[0]
    expected = torch.tensor([2.5, .5], dtype=ds.dtype, device=device) / (2 + 2 ** -.5)
    assert torch.allclose(aggregate(ds, 'fltrust', None, root), expected, atol=1e-12, rtol=0)
    scaled = ds.clone(); scaled[3] *= 1000
    assert torch.allclose(aggregate(scaled, 'fltrust', None, root), expected, atol=1e-12, rtol=0)
    assert torch.equal(aggregate(ds, 'fltrust', None, root * 0), root * 0)
    assert torch.equal(aggregate(-torch.ones_like(ds), 'fltrust', None, root), root * 0)
    gen = torch.Generator(device=device).manual_seed(118)
    ds = torch.randn((6, 19), generator=gen, dtype=torch.float64, device=device)
    root = ds.mean(0) + .25
    weights = [1 / 6] * 6
    config = dict(radius_multiplier=3., steps=8, step_radius_fraction=.25)
    checks = []
    for method in ['fedavg', 'trimmed_mean_f1', 'fltrust', 'fltrust_server']:
        v = ds[0].clone().requires_grad_(True)
        direction = aggregate(ds, method, weights, root).sign()
        def f(q):
            return (direction * aggregate(torch.cat([q[None], ds[1:]], 0), method, weights, root)).mean()
        grad, = torch.autograd.grad(f(v), v)
        for index in [0, 4, 11]:
            offset = torch.zeros_like(v); offset[index] = 1e-6
            numeric = (f(v.detach() + offset) - f(v.detach() - offset)) / 2e-6
            assert torch.allclose(numeric, grad[index], rtol=2e-4, atol=1e-8), (method, index)
        attack, info = adaptive(ds, method, weights, root, 0, config)
        checks.append(dict(method=method, improved=info['final_objective'] >= info['initial_best'], gradient_checks=3))
        if method == 'fedavg':
            radius = info['radius']
            optimum = -direction / torch.linalg.vector_norm(direction) * radius
            assert torch.allclose(attack, optimum, atol=1e-10, rtol=1e-10)
    return dict(passed=True, finite_difference_checks=12, fedavg_exact_constrained_optimum=True, methods=checks)


def quantile(s, w, q=.99):
    order = np.argsort(s, kind='stable')
    return float(s[order[min(np.searchsorted(np.cumsum(w[order]), q * w.sum()), len(s) - 1)]])


def confusion(y, decision, weight):
    c = np.bincount(2 * y.astype(int) + decision.astype(int), weights=weight, minlength=4)
    tn, fp, fn, tp = map(float, c)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=tp / (tp + fn) if tp + fn else None,
                fpr=fp / (fp + tn) if fp + tn else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixtures', action='store_true')
    for key in ['data', 'protocol', 'output']:
        ap.add_argument('--' + key, type=Path)
    ap.add_argument('--method'); ap.add_argument('--condition'); ap.add_argument('--seed', type=int, default=17); ap.add_argument('--arch', default='A', choices=['A', 'B'])
    args = ap.parse_args()
    if args.fixtures:
        print(json.dumps(fixtures())); return
    p = json.loads(args.protocol.read_text()); t = p['training']
    m = json.loads((args.data / 'manifest.json').read_text())
    assert args.method in p['methods'] and args.condition in p['conditions'] and args.seed in t['seeds']
    assert args.method != 'root_only' or args.condition == 'clean'
    arrays = {}
    for role in ['train', 'cal', 'period_a', 'period_b']:
        f = args.data / (role + '.npz'); assert digest(f) == m['files'][f.name]
        with np.load(f, allow_pickle=False) as z:
            arrays[role] = {k: z[k] for k in z.files}
    assert digest(args.data / 'partition.npz') == m['files']['partition.npz']
    with np.load(args.data / 'partition.npz') as z:
        root_ids, private = z['root'], z['private']
    train = arrays['train']
    clients = [private[train['owner'][private] == o] for o in m['owners']]
    assert len(root_ids) == 384 and all(len(i) for i in clients)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    free, _ = torch.cuda.mem_get_info(); assert free >= 3 * 1024 ** 3
    torch.cuda.set_per_process_memory_fraction(.08)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    fixture = fixtures('cuda')
    # Fixtures use a private Generator and do not advance model initialization RNG.
    if args.arch == 'A':
        model = nn.Sequential(nn.Linear(m['input_dim'], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
    else:
        model = nn.Sequential(nn.Linear(m['input_dim'], 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
    x = torch.from_numpy(train['x']).cuda(); y = torch.from_numpy(train['y'].astype(np.float32)).cuda()
    target = y.clone()
    attacker_name = None if args.condition == 'clean' else m['attackers'][0 if args.condition.endswith('_a') else 1]
    attacker = None if attacker_name is None else m['owners'].index(attacker_name)
    if args.condition.startswith('scaled_flip'):
        ids = torch.from_numpy(clients[attacker]).cuda(); target[ids] = 1 - y[ids]
    assert torch.equal(target[torch.from_numpy(root_ids).cuda()], y[torch.from_numpy(root_ids).cuda()])
    rngs = [np.random.default_rng(m['rng_base'] + 100 * args.seed + i) for i in range(len(clients) + 1)]
    lossfn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=t['learning_rate'], momentum=0)
    started = time.monotonic()
    result = dict(completed=False, dataset=m['dataset'], method=args.method, condition=args.condition, arch=args.arch,
                  attacker=attacker_name, attacker_weight=None if attacker is None else m['weights'][attacker],
                  seed=args.seed, protocol_sha256=digest(args.protocol), runner_sha256=digest(Path(__file__)),
                  manifest_sha256=digest(args.data / 'manifest.json'), input_sha256=m['files'],
                  confirmation_data_read=False, fixtures=fixture, rounds=[], checkpoints=[], probes=[],
                  environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))

    def save():
        result['seconds'] = time.monotonic() - started
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    def predict(a):
        output = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(a['x']), 8192):
                output.append(model(torch.from_numpy(a['x'][i:i + 8192]).cuda()).squeeze(-1).cpu().numpy())
        return np.concatenate(output)

    def checkpoint(step):
        score = {role: predict(a) for role, a in arrays.items() if role != 'train'}
        assert all(np.isfinite(s).all() for s in score.values())
        cal = arrays['cal']; benign = cal['y'] == 0
        threshold = quantile(score['cal'][benign], cal['weight'][benign])
        stats = {}
        for role, s in score.items():
            a = arrays[role]
            np.savez_compressed(args.output / f'{role}-step{step}.npz', score=s,
                                fixed=(s >= 0).astype(np.uint8), historical=(s > threshold).astype(np.uint8))
            stats[role] = dict(fixed=confusion(a['y'], s >= 0, a['weight']), historical=confusion(a['y'], s > threshold, a['weight']))
        result['checkpoints'].append(dict(step=step, threshold=threshold, labels_inspected=len(cal['y']), metrics=stats))
        save()
        print(json.dumps(dict(dataset=m['dataset'], method=args.method, condition=args.condition, step=step, seconds=time.monotonic() - started)), flush=True)

    def local(vector, ids, rng):
        with torch.no_grad():
            vector_to_parameters(vector.clone(), model.parameters())
        model.train()
        for _ in range(t['local_steps']):
            batch = torch.from_numpy(rng.choice(ids, size=min(t['batch_size'], len(ids)), replace=False)).cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = lossfn(model(x[batch]).squeeze(-1), target[batch]); loss.backward(); optimizer.step()
        return parameters_to_vector(model.parameters()).detach() - vector

    def probe(vector, ds, root, step):
        with torch.no_grad():
            vector_to_parameters(vector.clone(), model.parameters())
        group = {}
        for label in [0, 1]:
            ids = root_ids[train['y'][root_ids] == label]
            if not len(ids):
                group[str(label)] = dict(rows=0); continue
            ids = torch.from_numpy(ids).cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = lossfn(model(x[ids]).squeeze(-1), y[ids])
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            negative_gradient = -torch.cat([g.flatten() for g in grads]).double()
            vectors = torch.cat([ds, root[None]], dim=0).double()
            cos = vectors @ negative_gradient / (torch.linalg.vector_norm(vectors, dim=1) * torch.linalg.vector_norm(negative_gradient)).clamp_min(1e-30)
            group[str(label)] = dict(rows=len(ids), loss=float(loss.detach()), negative_gradient_norm=float(torch.linalg.vector_norm(negative_gradient)), client_then_root_cosines=cos.tolist())
        result['probes'].append(dict(step=step, root_label_groups=group))

    checkpoint(0)
    for step in range(1, t['rounds'] + 1):
        vector = parameters_to_vector(model.parameters()).detach().clone()
        ds = torch.stack([local(vector, ids, rngs[i]) for i, ids in enumerate(clients)]) if args.method != 'root_only' else torch.zeros((len(clients), len(vector)), device='cuda')
        root = local(vector, root_ids, rngs[-1])
        if not torch.isfinite(ds).all() or not torch.isfinite(root).all():
            result['failure'] = dict(step=step, stage='local_update'); save(); return
        clean_norms = torch.linalg.vector_norm(ds.double(), dim=1).tolist()
        if step in [1, 100, 300, 600]:
            probe(vector, ds, root, step)
        attack_info = None
        if args.condition.startswith('scaled_flip'):
            ds[attacker] /= m['weights'][attacker]
        elif args.condition.startswith('adaptive_deviation'):
            submitted, attack_info = adaptive(ds, args.method, m['weights'], root, attacker, p['adaptive_attack'])
            ds[attacker] = submitted
        delta = aggregate(ds, args.method, m['weights'], root)
        nxt = vector + t['global_learning_rate'] * delta
        if not torch.isfinite(nxt).all():
            result['failure'] = dict(step=step, stage='aggregate'); save(); return
        result['rounds'].append(dict(step=step, root_norm=float(torch.linalg.vector_norm(root.double())),
                                     local_update_norms=clean_norms, submitted_norms=torch.linalg.vector_norm(ds.double(), dim=1).tolist(),
                                     trust=trust_values(ds, root).tolist(), aggregate_norm=float(torch.linalg.vector_norm(delta.double())),
                                     adaptive=attack_info))
        with torch.no_grad():
            vector_to_parameters(nxt, model.parameters())
        if step in t['checkpoints']:
            checkpoint(step)
    torch.save(model.state_dict(), args.output / 'model-final.pt')
    result.update(completed=True, model_sha256=digest(args.output / 'model-final.pt'),
                  max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  score_sha256={f.name: digest(f) for f in sorted(args.output.glob('*-step*.npz'))})
    save()


if __name__ == '__main__':
    main()
