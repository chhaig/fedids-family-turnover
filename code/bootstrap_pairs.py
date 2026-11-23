"""Flow-level paired bootstrap for the 30 NF-ToN-IoT-v3 matched FLTrust/trimmed-mean pairs.
Resamples benign and attack rows (with their weights) within each period, identically for both
methods of a pair; recomputes oracle recall at 1% weighted FPR and AUC differences."""
import json, os, numpy as np, collections
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
LB = Path(__file__).resolve().parents[1] / 'labels'
B = int(os.environ.get('B', 400)); rng = np.random.default_rng(20261121)
U = json.load(open('unified_rescore.json'))
runs = {}
for o in U:
    if o['matrix'] == 'nf-root-seed-v1-matrix': runs.setdefault(o['run'], o)
pairs = collections.defaultdict(dict)
for r, o in runs.items(): pairs[(o['condition'], o['seed'], r.split('-')[-1])][o['method']] = o
lab = {}
def L(h):
    if h not in lab:
        z = np.load(LB / (h[:16] + '.npz')); lab[h] = (z['y'].astype(int), z['weight'].astype(float))
    return lab[h]
def orc(sb, wb, sa, wa):
    o = np.argsort(sb); c = np.cumsum(wb[o]); thr = sb[o][min(np.searchsorted(c, 0.99 * c[-1]), len(sb) - 1)]
    return (wa * (sa > thr)).sum() / wa.sum()
def auc(sb, wb, sa, wa):
    s = np.concatenate([sb, sa]); w = np.concatenate([wb, wa]); y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
    u, inv = np.unique(s, return_inverse=True)
    n = np.bincount(inv, w * (y == 0), len(u)); p = np.bincount(inv, w * (y == 1), len(u))
    return (p * (np.cumsum(n) - n + .5 * n)).sum() / (p.sum() * n.sum())
out = json.load(open('bootstrap_pairs.json')) if os.path.exists('bootstrap_pairs.json') else []
done = {(r['cond'], r['seed'], r['root']) for r in out}
import time; t0 = time.time()
for k, v in sorted(pairs.items()):
    if k in done: continue
    if time.time() - t0 > 140: break
    import zlib; rng = np.random.default_rng(zlib.crc32(repr(k).encode()))
    rec = dict(cond=k[0], seed=k[1], root=k[2])
    for per in ('period_a', 'period_b'):
        S = {}
        for m in ('fltrust', 'trimmed_mean_f1'):
            o = v[m]; d = EV / o['matrix'] / o['run']
            h = json.load(open(d / 'result.json'))['input_sha256'][per + '.npz']
            S[m] = np.load(d / f'{per}-step600.npz')['score'].astype(float)
        y, w = L(h); ib = np.flatnonzero(y == 0); ia = np.flatnonzero(y == 1)
        diffs = []; adiffs = []
        for b in range(B):
            bb = rng.choice(ib, len(ib)); aa = rng.choice(ia, len(ia))
            r = [orc(S[m][bb], w[bb], S[m][aa], w[aa]) for m in S]; diffs.append(r[0] - r[1])
            if b < 100:
                q = [auc(S[m][bb], w[bb], S[m][aa], w[aa]) for m in S]; adiffs.append(q[0] - q[1])
        rec[per] = dict(lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                        auc_lo=float(np.percentile(adiffs, 2.5)), auc_hi=float(np.percentile(adiffs, 97.5)))
    out.append(rec); json.dump(out, open('bootstrap_pairs.json', 'w'), indent=1)
    print(k, {p: (round(100*rec[p]['lo'],2), round(100*rec[p]['hi'],2)) for p in ('period_a','period_b')}, flush=True)
print('done')
