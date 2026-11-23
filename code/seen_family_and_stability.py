"""(1) Seen-family control: recall on calibration-evening DDoS (a training family) at the 1% FPR threshold
of the calibration benign flows, for the NF-ToN-IoT-v3 matched cells (FLTrust, trimmed mean, FedAvg).
(2) Mix-stability profile: over attack-family mixes drawn uniformly from the simplex, the share of mixes
where rule A leads by >1 pp, rule B leads by >1 pp, or neither (tie). Recall gap is linear in the mix."""
import json, os, numpy as np, collections, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[1]; LB = R / 'labels'; rng = np.random.default_rng(20261122)
labs = {}
def L(h):
    if h not in labs:
        z = np.load(LB / (h[:16] + '.npz')); labs[h] = (z['y'].astype(int), z['weight'].astype(float), z['family'])
    return labs[h]
def thr(s, w):
    o = np.argsort(s); c = np.cumsum(w[o]); return s[o][min(np.searchsorted(c, .99 * c[-1]), len(s) - 1)]
def auc(s, y, w):
    u, inv = np.unique(s, return_inverse=True); n = np.bincount(inv, w * (y == 0), len(u)); p = np.bincount(inv, w * (y == 1), len(u))
    return float((p * (np.cumsum(n) - n + .5 * n)).sum() / (p.sum() * n.sum()))
cells = collections.defaultdict(dict)
for base in (R / 'scores' / 'nf-root-seed-v1-matrix', R / 'scores' / 'fedavg-matched-new'):
    for d in sorted(base.iterdir()):
        r = json.load(open(d / 'result.json')); y, w, f = L(r['input_sha256']['cal.npz'])
        s = np.load(d / 'cal-step600.npz')['score'].astype(float); t = thr(s[y == 0], w[y == 0]); m = (y == 1) & (f == 'ddos')
        cells[(r['condition'], r['seed'], d.name.split('-')[-1])][r['method']] = dict(ddos=float((w[m] * (s[m] > t)).sum() / w[m].sum()), auc=auc(s, y, w))
seen = [dict(cond=k[0], seed=k[1], root=k[2], **v) for k, v in sorted(cells.items()) if len(v) == 3]
f1 = lambda x: round(100 * x, 2)
print('cells', len(seen))
for m in ('fltrust', 'trimmed_mean_f1', 'fedavg'):
    v = [c[m]['ddos'] for c in seen]; a = [c[m]['auc'] for c in seen]
    print(m, 'ddos recall min/med/max', f1(min(v)), f1(st.median(v)), f1(max(v)), 'auc min', round(min(a), 4))
g = [c['fltrust']['ddos'] - c['trimmed_mean_f1']['ddos'] for c in seen]
print('FLT-TM gap on seen DDoS: within 1pp', sum(abs(x) <= .01 for x in g), 'FLT>TM', sum(x > .01 for x in g), 'TM>FLT', sum(x < -.01 for x in g), 'range', f1(min(g)), f1(max(g)))
# (2) stability profile
def profile(d, n=20000):
    d = np.asarray(d); P = rng.dirichlet(np.ones(len(d)), n); D = P @ d
    return float((D > .01).mean()), float((np.abs(D) <= .01).mean()), float((D < -.01).mean())
FD = json.load(open('family_decomp.json')); prof = {}
for bp in ('period_a', 'period_b'):
    rows = [profile([r['d'][bp][k] for k in ('password', 'xss', 'Backdoor')]) for r in FD]
    prof['nf_' + bp] = rows
    a = np.array(rows); print('NF', bp, 'median shares FLT/tie/TM', np.round(np.median(a, 0), 3), 'pairs where both rules lead on >=10% of mixes', int(((a[:, 0] >= .1) & (a[:, 2] >= .1)).sum()), 'pairs with one rule leading on >=90%', int(((a[:, 0] >= .9) | (a[:, 2] >= .9)).sum()))
U = json.load(open('unified_rescore.json')); P = collections.defaultdict(dict); seenr = set()
for o in U:
    if o['matrix'] == 'nf-unsw-study-matrix-v1' and o['run'] not in seenr and o['method'] in ('fltrust', 'trimmed_mean_f1'):
        seenr.add(o['run']); P[(o['condition'], o['seed'])][o['method']] = o
rows = []
for k, v in P.items():
    if len(v) == 2:
        fams = sorted(v['fltrust']['period_b']['family']); rows.append(profile([v['fltrust']['period_b']['family'][f][2] - v['trimmed_mean_f1']['period_b']['family'][f][2] for f in fams]))
prof['unsw_period_b'] = rows; a = np.array(rows)
print('UNSW later', len(rows), 'median shares FLT/tie/TM', np.round(np.median(a, 0), 3), 'pairs where both lead on >=10%', int(((a[:, 0] >= .1) & (a[:, 2] >= .1)).sum()), 'min tie share', round(float(a[:, 1].min()), 3))
json.dump(dict(seen_family=seen, stability=prof), open('seen_family_and_stability.json', 'w'), indent=1)
