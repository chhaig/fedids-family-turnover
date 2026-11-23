"""Benign-pool x attack-pool factorial on saved scores: oracle recall at 1% weighted FPR
when attacks of period X are ranked against benign flows of period Y."""
import json, glob, os, numpy as np, collections
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
LB = Path(__file__).resolve().parents[1] / 'labels'
def wq(s, w, q):
    o = np.argsort(s); c = np.cumsum(w[o]) / w.sum(); return s[o][min(np.searchsorted(c, q), len(s) - 1)]
cache = {}
def lab(h):
    if h not in cache:
        z = np.load(LB / (h[:16] + '.npz')); cache[h] = {k: z[k] for k in ('y', 'weight', 'family')}
    return cache[h]
U = json.load(open('unified_rescore.json'))
want = {('nf-root-seed-v1-matrix'), ('cross-testbed-adaptive-v1-matrix'), ('strong-core-baseline-v1-matrix')}
seen = set(); out = []
for o in U:
    if o['matrix'] not in want or (o['matrix'], o['run']) in seen: continue
    if o['method'] not in ('fltrust', 'trimmed_mean_f1'): continue
    seen.add((o['matrix'], o['run']))
    d = EV / o['matrix'] / o['run']
    r = json.load(open(d / 'result.json')); ish = r['input_sha256']
    S = {}; 
    for p in ('period_a', 'period_b'):
        L = lab(ish[p + '.npz']); s = np.load(d / f'{p}-step600.npz')['score'].astype(float)
        S[p] = dict(ben=(s[L['y'] == 0], L['weight'][L['y'] == 0]), att=(s[L['y'] == 1], L['weight'][L['y'] == 1]))
    res = {}
    for bp in S:
        thr = wq(*S[bp]['ben'], 0.99)
        for ap in S:
            sa, wa = S[ap]['att']; res[f'ben_{bp[-1]}|att_{ap[-1]}'] = float(wa[sa > thr].sum() / wa.sum())
    out.append(dict(matrix=o['matrix'], run=o['run'], dataset=o['dataset'], method=o['method'], condition=o['condition'], seed=o['seed'], swap=res))
json.dump(out, open('pool_swap.json', 'w'), indent=1)
print(len(out))
