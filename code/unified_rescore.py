"""Unified re-scoring of saved federated-IDS scores (step 600) against exported labels.
Per run and period: weighted AUC, same-period oracle recall at 1% FPR, historical-threshold
recall/FPR (checked against result.json), per-family recall, per-owner FPR with benign weight."""
import json, glob, os, sys, numpy as np
from pathlib import Path
EV = Path(__file__).resolve().parents[1] / 'scores'
LB = Path(__file__).resolve().parents[1] / 'labels'
labels = {p.stem: p for p in LB.glob('*.npz')}
cache = {}
def lab(h):
    k = h[:16]
    if k not in labels: return None
    if k not in cache:
        z = np.load(labels[k], allow_pickle=False); cache[k] = {n: z[n] for n in z.files}
    return cache[k]
def wquant(s, w, q):
    o = np.argsort(s); s, w = s[o], w[o]; c = np.cumsum(w) / w.sum()
    return s[min(np.searchsorted(c, q), len(s) - 1)]
def wauc(s, y, w):
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    uniq, inv = np.unique(s, return_inverse=True)
    n = np.bincount(inv, weights=w * (y == 0), minlength=len(uniq)); p = np.bincount(inv, weights=w * (y == 1), minlength=len(uniq))
    cn = np.cumsum(n) - n
    wp, wn = p.sum(), n.sum()
    return float((p * (cn + 0.5 * n)).sum() / (wp * wn)) if wp > 0 and wn > 0 else float('nan')
def rates(dec, y, w):
    P = w[y == 1].sum(); N = w[y == 0].sum()
    return (float(w[(y == 1) & dec].sum() / P) if P else float('nan'), float(w[(y == 0) & dec].sum() / N) if N else float('nan'))
out = []
for rj in sorted(glob.glob(str(EV / '*/*/result.json'))):
    if Path(rj).parent.parent.name in ('seed-panel-new', 'fedavg-matched-new'): continue
    try: r = json.load(open(rj))
    except Exception: continue
    if not (r.get('completed') or 'final_checkpoint' in r) or 'input_sha256' not in r: continue
    ish = r['input_sha256']; d = Path(rj).parent
    L = {k: lab(ish.get(k + '.npz', '')) for k in ['cal', 'period_a', 'period_b']}
    if any(v is None for v in L.values()) or not (d / 'cal-step600.npz').exists(): continue
    sc = {k: np.load(d / f'{k}-step600.npz')['score'].astype(np.float64) for k in L}
    if any(len(sc[k]) != len(L[k]['y']) for k in L): continue
    cy, cw = L['cal']['y'].astype(int), L['cal']['weight'].astype(np.float64)
    thr = wquant(sc['cal'][cy == 0], cw[cy == 0], 0.99)
    rec = dict(matrix=d.parent.name, run=d.name, node='local', dataset=r['dataset'], method=r['method'], condition=r['condition'], seed=r['seed'],
               inputs={k: ish[k + '.npz'][:16] for k in L}, threshold=float(thr))
    for k in ['period_a', 'period_b']:
        y = L[k]['y'].astype(int); w = L[k]['weight'].astype(np.float64); s = sc[k]
        oth = wquant(s[y == 0], w[y == 0], 0.99)
        hr, hf = rates(s > thr, y, w); orc, ofp = rates(s > oth, y, w)
        checkpoint = r['final_checkpoint'] if 'final_checkpoint' in r else r['checkpoints'][-1]
        saved = checkpoint['metrics'][k]['historical']
        fam = {}
        for f in np.unique(L[k]['family'][y == 1]):
            m = (y == 1) & (L[k]['family'] == f)
            fam[str(f)] = [float(w[m].sum()), float(w[m & (s > thr)].sum() / w[m].sum()), float(w[m & (s > oth)].sum() / w[m].sum())]
        own = {}
        for o in np.unique(L[k]['owner']):
            m = (y == 0) & (L[k]['owner'] == o)
            own[str(o)] = [float(w[m].sum()), int(m.sum()), float(w[m & (s > thr)].sum() / w[m].sum()) if w[m].sum() else None]
        rec[k] = dict(auc=wauc(s, y, w), oracle_recall=orc, oracle_fpr=ofp, hist_recall=hr, hist_fpr=hf,
                      saved_hist_recall=saved['recall'], saved_hist_fpr=saved['fpr'], family=fam, owner=own,
                      days=sorted(set(map(str, np.unique(L[k]['day'])))))
    out.append(rec)
json.dump(out, open('unified_rescore.json', 'w'), indent=1)
bad = [o for o in out for k in ['period_a', 'period_b'] if abs(o[k]['hist_recall'] - o[k]['saved_hist_recall']) > 1e-6 or abs(o[k]['hist_fpr'] - o[k]['saved_hist_fpr']) > 1e-6]
print('runs', len(out), 'mismatch vs saved', len(bad))
from collections import Counter; print(Counter((o['matrix'], o['dataset']) for o in out))
