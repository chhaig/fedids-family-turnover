#!/usr/bin/env python3
"""NF-CSE-CIC-IDS2018-v3 family-turnover panel on node 226 with the archived cross-testbed runner (unchanged).
90 runs: fltrust / trimmed_mean_f1 / fedavg x clean / scaled_flip_a / scaled_flip_b x seeds 29, 43 x five root draws.
Outputs only under FEDIDS_WORK_DIR/fedids_cic18. Two sequential background workers, each with the runner's own
8% GPU memory cap and >=3 GiB free check; launched only if >=8 GiB GPU memory is free. No other process is touched.
First call installs data and launches; later calls report status and copy compact step-600 scores to exchange/."""
import json, os, shutil, subprocess, time, glob
from pathlib import Path
# Explicit local paths are required; this script never connects to a server.
D = Path(os.environ['FEDIDS_WORK_DIR'])
SRC = Path(os.environ['FEDIDS_CROSS_TESTBED_DIR'])
OUT = D / 'fedids_cic18'; OUT.mkdir(parents=True, exist_ok=True)
EX = D / 'exchange' / 'fedids_cic18'; EX.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'data'; rep = {'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
runner = SRC / 'cross_testbed_runner_v1.py'; proto = SRC / 'cross-testbed-adaptive-v1.json'
py = Path(os.environ['FEDIDS_TRAIN_PYTHON'])
rep.update(runner=runner.exists(), protocol=proto.exists(), python=py.exists())
inbox = D / 'exchange_in' / 'cic18'
if inbox.exists() and not (DATA / 'base' / 'manifest-base.json').exists():
    (DATA / 'base').mkdir(parents=True, exist_ok=True)
    for f in inbox.iterdir(): shutil.copy2(f, DATA / 'base' / f.name)
    rep['installed_data'] = sorted(p.name for p in (DATA / 'base').iterdir())
base = DATA / 'base'; draws = []
if (base / 'manifest-base.json').exists():
    mb = json.loads((base / 'manifest-base.json').read_text())
    import hashlib
    def digest(p):
        h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
    for s, info in sorted(mb['root_draws'].items()):
        dd = DATA / f'draw{s}' / 'cic18'
        if not (dd / 'manifest.json').exists():
            dd.mkdir(parents=True, exist_ok=True)
            for r in ['train', 'cal', 'period_a', 'period_b', 'seen']:
                if not (dd / f'{r}.npz').exists(): os.symlink(base / f'{r}.npz', dd / f'{r}.npz')
            shutil.copy2(base / info['file'], dd / 'partition.npz')
            m = {k: v for k, v in mb.items() if k not in ('cells', 'root_draws')}
            m['files'] = {f: mb['files'][f] for f in ['train.npz', 'cal.npz', 'period_a.npz', 'period_b.npz', 'seen.npz']}
            m['files']['partition.npz'] = digest(dd / 'partition.npz'); m['root_draw'] = int(s)
            assert all(digest(dd / f) == h for f, h in m['files'].items())
            (dd / 'manifest.json').write_text(json.dumps(m, indent=1))
        draws.append((s, dd))
rep['draws'] = [s for s, _ in draws]
proto2 = OUT / 'protocol-seeds.json'
if proto.exists() and not proto2.exists():
    p = json.loads(proto.read_text()); p['training']['seeds'] = [17, 29, 43]
    p['amendment'] = 'Copy of cross-testbed-adaptive-v1 with seeds 29 and 43 for the NF-CSE-CIC-IDS2018-v3 family-turnover panel; all other fields unchanged.'
    proto2.write_text(json.dumps(p, indent=1))
jobs = [(s, dd, m, c, sd) for s, dd in draws for sd in (29, 43) for c in ('clean', 'scaled_flip_a', 'scaled_flip_b') for m in ('fltrust', 'trimmed_mean_f1', 'fedavg')]
name = lambda s, m, c, sd: f'cic18-{m}-{c}-{sd}-draw{s}'
smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
rep['gpu'] = smi
SCORE = OUT / 'score_seen.py'
SCORE.write_text('''import sys, numpy as np, torch
from torch import nn
run, data = sys.argv[1], sys.argv[2]
z = np.load(data + '/seen.npz'); x = torch.from_numpy(z['x'])
m = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
m.load_state_dict(torch.load(run + '/model-final.pt', map_location='cpu')); m.eval()
with torch.no_grad(): s = torch.cat([m(x[i:i + 8192]).squeeze(-1) for i in range(0, len(x), 8192)]).numpy()
np.savez_compressed(run + '/seen-step600.npz', score=s)
''')
marker = OUT / 'launched'
free_ok = False
if smi:
    used, total, _ = [float(v) for v in smi.splitlines()[0].split(',')]; free_ok = (total - used) >= 8192
rep['free_ok'] = free_ok
if runner.exists() and py.exists() and len(draws) == 5 and free_ok and not marker.exists():
    for w in range(2):
        lines = ['#!/bin/bash', 'cd ' + str(runner.parent)]
        for i, (s, dd, m, c, sd) in enumerate(jobs):
            if i % 2 != w: continue
            o = OUT / name(s, m, c, sd)
            lines.append(f'[ -f {o}/seen-step600.npz ] || {{ rm -rf {o}; {py} {runner} --data {dd} --protocol {proto2} --output {o} --method {m} --condition {c} --seed {sd} > {o}.log 2>&1 && {py} {SCORE} {o} {dd} >> {o}.log 2>&1; }}')
        lines.append(f'echo DONE > {OUT}/worker{w}.done')
        q = OUT / f'queue{w}.sh'; q.write_text('\n'.join(lines) + '\n'); q.chmod(0o755)
        subprocess.Popen(['setsid', 'nohup', 'bash', str(q)], stdout=open(OUT / f'queue{w}.out', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    marker.write_text(rep['time']); rep['launched'] = True
import numpy as np
done = []
for s, dd, m, c, sd in jobs:
    o = OUT / name(s, m, c, sd); e = EX / o.name
    if (o / 'seen-step600.npz').exists():
        done.append(o.name)
        if not (e / 'scores.npz').exists():
            e.mkdir(exist_ok=True)
            sc = {r: np.load(o / f'{r}-step600.npz')['score'].astype(np.float32) for r in ['cal', 'period_a', 'period_b', 'seen']}
            np.savez_compressed(e / 'scores.npz', **sc)
            j = json.loads((o / 'result.json').read_text())
            keep = {k: j[k] for k in j if k not in ('rounds', 'probes')}
            keep['last_round'] = j['rounds'][-1] if j.get('rounds') else None
            (e / 'result.json').write_text(json.dumps(keep, indent=1))
    elif (o / 'result.json').exists():
        try:
            j = json.loads((o / 'result.json').read_text())
            if j.get('failure'): rep.setdefault('failures', {})[o.name] = j['failure']
        except Exception: pass
rep['done'] = len(done); rep['total'] = len(jobs); rep['workers_done'] = [p.name for p in OUT.glob('worker*.done')]
logs = sorted(OUT.glob('*.log'), key=os.path.getmtime)
if logs: rep['last_log'] = logs[-1].name + ': ' + logs[-1].read_text()[-400:]
(EX / 'status.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
