"""Operating-point sensitivity for the 30 NF-ToN-IoT-v3 matched pairs: oracle recall gap at
target FPR in {0.1,0.5,1,2,5}% per period, and family opposite-sign pattern with later benign pool."""
import json, os, numpy as np, collections
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
LB = Path(__file__).resolve().parents[1] / 'labels'
U = json.load(open('unified_rescore.json')); runs = {}
for o in U:
    if o['matrix'] == 'nf-root-seed-v1-matrix': runs.setdefault(o['run'], o)
pairs = collections.defaultdict(dict)
for r, o in runs.items(): pairs[(o['condition'], o['seed'], r.split('-')[-1])][o['method']] = o
labs = {}
def L(h):
    if h not in labs:
        z = np.load(LB / (h[:16] + '.npz')); labs[h] = (z['y'].astype(int), z['weight'].astype(float), z['family'])
    return labs[h]
def thr(s, w, q):
    o = np.argsort(s); c = np.cumsum(w[o]); return s[o][min(np.searchsorted(c, q * c[-1]), len(s) - 1)]
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05]; out = []
for k, v in sorted(pairs.items()):
    S = {}
    for m in ('fltrust', 'trimmed_mean_f1'):
        o = v[m]; d = EV / o['matrix'] / o['run']
        ish = json.load(open(d / 'result.json'))['input_sha256']
        S[m] = {p: (np.load(d / f'{p}-step600.npz')['score'].astype(float), L(ish[p + '.npz'])) for p in ('period_a', 'period_b')}
    rec = dict(cond=k[0], seed=k[1], root=k[2], fpr={})
    for a in FPRS:
        e = {}
        for m in S:
            for p in ('period_a', 'period_b'):
                s, (y, w, f) = S[m][p]; t = thr(s[y == 0], w[y == 0], 1 - a)
                e[(m, p)] = (w * (s > t) * (y == 1)).sum() / w[y == 1].sum()
            # families at later-benign threshold
            sb, (yb, wb, fb) = S[m]['period_b']; t = thr(sb[yb == 0], wb[yb == 0], 1 - a)
            sa, (ya, wa, fa) = S[m]['period_a']
            e[(m, 'pw')] = (wa * (sa > t) * (fa == 'password') * (ya == 1)).sum() / wa[(ya == 1) & (fa == 'password')].sum()
            e[(m, 'bd')] = (wb * (sb > t) * (fb == 'Backdoor') * (yb == 1)).sum() / wb[(yb == 1) & (fb == 'Backdoor')].sum()
        g = lambda x: float(e[('fltrust', x)] - e[('trimmed_mean_f1', x)])
        rec['fpr'][str(a)] = dict(dA=g('period_a'), dB=g('period_b'), dpw=g('pw'), dbd=g('bd'))
    out.append(rec)
json.dump(out, open('op_sensitivity.json', 'w'), indent=1)
import statistics as st
for a in FPRS:
    R = [r['fpr'][str(a)] for r in out]
    print(a, 'material reversal', sum(abs(r['dA']) > .01 and abs(r['dB']) > .01 and (r['dA'] > 0) != (r['dB'] > 0) for r in R),
          'FLT>TM earlier', sum(r['dA'] > .01 for r in R), 'TM>FLT later', sum(r['dB'] < -.01 for r in R),
          'family opposite (later benign)', sum(r['dpw'] > .01 and r['dbd'] < -.01 for r in R),
          'median dA', round(100 * st.median(r['dA'] for r in R), 1), 'median dB', round(100 * st.median(r['dB'] for r in R), 1))
