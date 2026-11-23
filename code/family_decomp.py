"""Family-composition decomposition on the NF-ToN-IoT-v3 matched panel.
With the benign pool held fixed, recall at a 1% FPR threshold is a mixture of per-family recalls:
R(pi) = sum_f pi_f r_f. The FLTrust-minus-trimmed-mean gap is D(pi) = sum_f pi_f d_f. We compute
per-family gaps d_f for password, XSS and Backdoor against each benign pool, the gap at each
period's own family mix, and the Backdoor share pi* at which the sign of D changes."""
import json, os, numpy as np, collections
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
LB = Path(__file__).resolve().parents[1] / 'labels'
U = json.load(open('unified_rescore.json'))
runs = {}
for o in U:
    if o['matrix'] == 'nf-root-seed-v1-matrix': runs.setdefault(o['run'], o)
pairs = collections.defaultdict(dict)
for r, o in runs.items(): pairs[(o['condition'], o['seed'], r.split('-')[-1])][o['method']] = o
labs = {}
def L(h):
    if h not in labs:
        z = np.load(LB / (h[:16] + '.npz')); labs[h] = (z['y'].astype(int), z['weight'].astype(float), z['family'])
    return labs[h]
def thr(s, w):
    o = np.argsort(s); c = np.cumsum(w[o]); return s[o][min(np.searchsorted(c, .99 * c[-1]), len(s) - 1)]
out = []
for k, v in sorted(pairs.items()):
    rec = dict(cond=k[0], seed=k[1], root=k[2], d={}, mix={})
    S = {}
    for m in ('fltrust', 'trimmed_mean_f1'):
        o = v[m]; d = EV / o['matrix'] / o['run']
        ish = json.load(open(d / 'result.json'))['input_sha256']
        S[m] = {p: (np.load(d / f'{p}-step600.npz')['score'].astype(float), L(ish[p + '.npz'])) for p in ('period_a', 'period_b')}
    for bp in ('period_a', 'period_b'):
        r = {}
        for m in S:
            s, (y, w, f) = S[m][bp]; t = thr(s[y == 0], w[y == 0])
            fr = {}
            for p in ('period_a', 'period_b'):
                s2, (y2, w2, f2) = S[m][p]
                for fam in ('password', 'xss', 'Backdoor'):
                    msk = (y2 == 1) & (f2 == fam)
                    if msk.any(): fr[fam] = (w2[msk] * (s2[msk] > t)).sum() / w2[msk].sum()
            r[m] = fr
        dfam = {fam: r['fltrust'][fam] - r['trimmed_mean_f1'][fam] for fam in r['fltrust']}
        rec['d'][bp] = dfam
        # earlier mix: password/xss weights; later: Backdoor
        _, (y, w, f) = S['fltrust']['period_a']
        wp = w[(y == 1) & (f == 'password')].sum(); wx = w[(y == 1) & (f == 'xss')].sum()
        dA = (wp * dfam['password'] + wx * dfam['xss']) / (wp + wx); dB = dfam['Backdoor']
        star = dA / (dA - dB) if (dA > 0) != (dB > 0) else None
        rec['mix'][bp] = dict(d_earlier_mix=dA, d_later_mix=dB, backdoor_share_at_crossover=star)
    out.append(rec)
json.dump(out, open('family_decomp.json', 'w'), indent=1)
f = lambda x: round(100 * x, 1)
for r in out:
    print(r['cond'], r['seed'], r['root'][-2:], {bp: {fam: f(x) for fam, x in r['d'][bp].items()} for bp in r['d']},
          {bp: (None if r['mix'][bp]['backdoor_share_at_crossover'] is None else round(r['mix'][bp]['backdoor_share_at_crossover'], 3)) for bp in r['mix']})
