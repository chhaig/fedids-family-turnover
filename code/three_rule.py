"""Three-rule matched comparison on NF-ToN-IoT-v3: FedAvg (new) vs FLTrust and trimmed mean (archived),
same root draw, seed and condition. Oracle recall at 1% FPR per period and per family (later benign pool)."""
import json, os, numpy as np, collections, itertools
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
NEW = Path(__file__).resolve().parents[1] / 'scores/fedavg-matched-new'
LB = Path(__file__).resolve().parents[1] / 'labels'
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
def score(d):
    r = json.load(open(d / 'result.json')); ish = r['input_sha256']; o = {}
    S = {p: (np.load(d / f'{p}-step600.npz')['score'].astype(float), L(ish[p + '.npz'])) for p in ('period_a', 'period_b')}
    for p in S:
        s, (y, w, f) = S[p]; t = thr(s[y == 0], w[y == 0])
        o[p] = float((w * (s > t) * (y == 1)).sum() / w[y == 1].sum()); o['auc_' + p] = auc(s, y, w)
    sb, (yb, wb, fb) = S['period_b']; t = thr(sb[yb == 0], wb[yb == 0]); sa, (ya, wa, fa) = S['period_a']
    for fam, (s, y, w, f) in {'password': (sa, ya, wa, fa), 'xss': (sa, ya, wa, fa), 'Backdoor': (sb, yb, wb, fb)}.items():
        m = (y == 1) & (f == fam); o[fam] = float((w[m] * (s[m] > t)).sum() / w[m].sum())
    return o
U = json.load(open('analysis/unified_rescore.json') if os.path.exists('analysis/unified_rescore.json') else open('unified_rescore.json'))
cells = collections.defaultdict(dict); seen = set()
for o in U:
    if o['matrix'] == 'nf-root-seed-v1-matrix' and o['run'] not in seen:
        seen.add(o['run']); cells[(o['condition'], o['seed'], o['run'].split('-')[-1])][o['method']] = score(EV / o['matrix'] / o['run'])
for d in sorted(NEW.glob('nf-fedavg-*')):
    if (d / 'result.json').exists():
        r = json.load(open(d / 'result.json'))
        if r.get('completed'): cells[(r['condition'], r['seed'], d.name.split('-')[-1])]['fedavg'] = score(d)
full = {k: v for k, v in cells.items() if len(v) == 3}
out = [dict(cond=k[0], seed=k[1], root=k[2], **{m: v[m] for m in v}) for k, v in sorted(full.items())]
json.dump(out, open('three_rule.json', 'w'), indent=1)
print('complete cells', len(full))
M = ['fedavg', 'fltrust', 'trimmed_mean_f1']
top = collections.Counter(); 
for k, v in full.items():
    a = max(M, key=lambda m: v[m]['period_a']); b = max(M, key=lambda m: v[m]['period_b']); top[('earlier', a)] += 1; top[('later', b)] += 1; top[('same_winner', a == b)] += 1
print(sorted(top.items(), key=str))
for x, y in itertools.combinations(M, 2):
    rev = sum(abs(v[x]['period_a'] - v[y]['period_a']) > .01 and abs(v[x]['period_b'] - v[y]['period_b']) > .01 and ((v[x]['period_a'] > v[y]['period_a']) != (v[x]['period_b'] > v[y]['period_b'])) for v in full.values())
    fam = sum(abs(v[x]['password'] - v[y]['password']) > .01 and abs(v[x]['Backdoor'] - v[y]['Backdoor']) > .01 and ((v[x]['password'] > v[y]['password']) != (v[x]['Backdoor'] > v[y]['Backdoor'])) for v in full.values())
    print(x, 'vs', y, 'material period reversal', rev, '| opposite sign password vs Backdoor (later benign pool)', fam)
import statistics as st
for m in M:
    print(m, {q: [round(100 * f([v[m][q] for v in full.values()]), 1) for f in (min, st.median, max)] for q in ('period_a', 'period_b', 'password', 'xss', 'Backdoor')})
