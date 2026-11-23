"""Seed replication on CIC-IDS2017 and IoT-23: FLTrust vs trimmed mean, seeds 17 (archived) + 29, 43 (new).
Per model and period: AUC, oracle recall at 1% FPR, historical-threshold FPR; per pair: material differences."""
import json, os, glob, numpy as np, collections
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
NEW = Path(__file__).resolve().parents[1] / 'scores/seed-panel-new'
LB = Path(__file__).resolve().parents[1] / 'labels'
def wq(s, w, q=.99):
    o = np.argsort(s); c = np.cumsum(w[o]); return s[o][min(np.searchsorted(c, q * c[-1]), len(s) - 1)]
def auc(s, y, w):
    u, inv = np.unique(s, return_inverse=True)
    n = np.bincount(inv, w * (y == 0), len(u)); p = np.bincount(inv, w * (y == 1), len(u))
    return float((p * (np.cumsum(n) - n + .5 * n)).sum() / (p.sum() * n.sum()))
def score(d):
    r = json.load(open(d / 'result.json')); ish = r['input_sha256']; out = dict(ds=r['dataset'], m=r['method'], c=r['condition'], s=r['seed'])
    z = np.load(LB / (ish['cal.npz'][:16] + '.npz')); cy, cw = z['y'], z['weight']; cs = np.load(d / 'cal-step600.npz')['score']
    thr = wq(cs[cy == 0], cw[cy == 0])
    for p in ('period_a', 'period_b'):
        z = np.load(LB / (ish[p + '.npz'][:16] + '.npz')); y, w, f = z['y'], z['weight'].astype(float), z['family']
        s = np.load(d / f'{p}-step600.npz')['score'].astype(float)
        o = wq(s[y == 0], w[y == 0])
        out[p] = dict(auc=auc(s, y, w), oracle=float((w * (s > o) * (y == 1)).sum() / w[y == 1].sum()),
                      hist_fpr=float((w * (s > thr) * (y == 0)).sum() / w[y == 0].sum()),
                      fam={str(k): float((w * (s > o) * ((y == 1) & (f == k))).sum() / w[(y == 1) & (f == k)].sum()) for k in np.unique(f[y == 1])})
    return out
rows = []
for d in sorted(NEW.glob('*-*-*-*')):
    if (d / 'result.json').exists() and json.load(open(d / 'result.json')).get('completed'): rows.append(score(d))
for mx in ('cross-testbed-adaptive-v1-matrix',):
    seen = set()
    for rj in sorted(glob.glob(str(EV / mx / '*-17/result.json'))):
        d = Path(rj).parent; nm = d.name
        if nm in seen: continue
        r = json.load(open(rj))
        if r['dataset'] in ('cic', 'iot') and r['method'] in ('fltrust', 'trimmed_mean_f1') and r['condition'] in ('clean', 'scaled_flip_a', 'scaled_flip_b'):
            seen.add(nm); rows.append(score(d))
json.dump(rows, open('seed_replication.json', 'w'), indent=1)
P = collections.defaultdict(dict)
for r in rows: P[(r['ds'], r['c'], r['s'])][r['m']] = r
f = lambda x: round(100 * x, 1)
for k in sorted(P):
    v = P[k]
    if len(v) < 2: continue
    a, b = v['fltrust'], v['trimmed_mean_f1']
    print(k, 'FLT A/B', f(a['period_a']['oracle']), f(a['period_b']['oracle']), 'TM A/B', f(b['period_a']['oracle']), f(b['period_b']['oracle']),
          'AUC B', round(a['period_b']['auc'], 3), round(b['period_b']['auc'], 3), 'histFPR B', f(a['period_b']['hist_fpr']), f(b['period_b']['hist_fpr']))
print('models', len(rows))
