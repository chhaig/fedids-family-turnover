"""Same-day seen-vs-unseen family test on NF-ToN-IoT-v3, 26 April 2019.
The development sample (cross_dev.npz) holds, for 26 April, benign flows, DDoS flows (a training family)
and password flows (absent from training). Each frozen model is run on these rows (numpy forward pass of
the saved MLP weights, checked against the archived scores); recall is computed per family at the 1% FPR
threshold of the 26 April benign flows."""
import json, os, sys, zipfile, pickle, numpy as np, collections, statistics as st
from pathlib import Path
P = Path(__file__).resolve().parents[1]
DEV = np.load(Path(os.environ.get('FEDIDS_DEV', str(P / 'external-data/cross_dev.npz'))))
X = DEV['x'].astype(np.float32); Y = DEV['y'].astype(int); F = DEV['family']; W = DEV['population_weight'].astype(float); DAY = DEV['end_ms'] // 86400000
LB = P / 'labels'
def load_pt(path):
    """Minimal reader for a torch state_dict zip (no torch needed)."""
    zf = zipfile.ZipFile(path); root = [n for n in zf.namelist() if n.endswith('data.pkl')][0][:-8]
    class S:  # storage stub
        def __init__(s, key, dtype): s.key, s.dtype = key, dtype
    def rebuild(storage, offset, size, stride, *a):
        arr = np.frombuffer(zf.read(root + 'data/' + storage.key), dtype=storage.dtype)
        n = int(np.prod(size)) if len(size) else 1
        return arr[offset:offset + n].reshape(size).copy()
    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild
            if name == 'OrderedDict': return collections.OrderedDict
            if name.endswith('Storage'): return {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64}[name]
            return super().find_class(mod, name)
        def persistent_load(s, pid): return S(pid[2], pid[1])
    return U(zf.open(root + 'data.pkl')).load()
def forward(sd, x):
    k = list(sd.values()); h = x
    for i in range(0, len(k), 2):
        h = h @ k[i].T.astype(np.float32) + k[i + 1].astype(np.float32)
        if i < len(k) - 2: h = np.maximum(h, 0)
    return h[:, 0]
def thr(s, w):
    o = np.argsort(s); c = np.cumsum(w[o]); return s[o][min(np.searchsorted(c, .99 * c[-1]), len(s) - 1)]
dirs = []
for base in sys.argv[1:]:
    dirs += sorted(d for d in Path(base).iterdir() if (d / 'model-final.pt').exists() and d.name.startswith('nf-'))
d26 = DAY == 18012; ben = d26 & (Y == 0); out = []; maxdiff = 0.0; checked = 0
print('26 April rows: benign', int(ben.sum()), {f: int((d26 & (F == f)).sum()) for f in ('ddos', 'password', 'xss')})
for d in dirs:
    r = json.load(open(d / 'result.json'))
    if r['method'] not in ('fltrust', 'trimmed_mean_f1', 'fedavg') or r['condition'] not in ('clean', 'scaled_flip_a', 'scaled_flip_b'): continue
    s = forward(load_pt(d / 'model-final.pt'), X).astype(np.float64)
    if (d / 'period_a-step600.npz').exists():
        ix = np.load(LB / (r['input_sha256']['period_a.npz'][:16] + '.npz'))['original_index']
        ref = np.load(d / 'period_a-step600.npz')['score']; maxdiff = max(maxdiff, float(np.abs(s[ix] - ref).max())); checked += 1
    t = thr(s[ben], W[ben]); rec = dict(run=d.name, method=r['method'], cond=r['condition'], seed=r['seed'], root=d.name.split('-')[-1])
    for f in ('ddos', 'password'):
        m = d26 & (F == f); rec[f] = float((W[m] * (s[m] > t)).sum() / W[m].sum())
    hod = (DEV['start_ms'] % 86400000) / 3600000
    for f, bm in (('ddos', ben & (hod < 6)), ('password', ben & (hod >= 6))):
        m = d26 & (F == f); tt = thr(s[bm], W[bm]); rec[f + '_tm'] = float((W[m] * (s[m] > tt)).sum() / W[m].sum())
    out.append(rec)
print('models', len(out), 'checked against archived scores', checked, 'max abs score difference', maxdiff)
json.dump(out, open('same_day_family.json', 'w'), indent=1)
cells = collections.defaultdict(dict)
for r in out: cells[(r['cond'], r['seed'], r['root'])][r['method']] = r
f1 = lambda x: round(100 * x, 2)
for m in ('fltrust', 'trimmed_mean_f1', 'fedavg'):
    v = [c[m] for c in cells.values() if m in c]
    if v: print(m, len(v), 'ddos min/med/max', f1(min(x['ddos'] for x in v)), f1(st.median(x['ddos'] for x in v)), f1(max(x['ddos'] for x in v)), '| password', f1(min(x['password'] for x in v)), f1(st.median(x['password'] for x in v)), f1(max(x['password'] for x in v)))
pairs = [c for c in cells.values() if 'fltrust' in c and 'trimmed_mean_f1' in c]
gd = [c['fltrust']['ddos'] - c['trimmed_mean_f1']['ddos'] for c in pairs]; gp = [c['fltrust']['password'] - c['trimmed_mean_f1']['password'] for c in pairs]
gd2 = [c['fltrust']['ddos_tm'] - c['trimmed_mean_f1']['ddos_tm'] for c in pairs]; gp2 = [c['fltrust']['password_tm'] - c['trimmed_mean_f1']['password_tm'] for c in pairs]
import numpy as _n
print('time-matched benign: ddos FLT', f1(min(c['fltrust']['ddos_tm'] for c in pairs)), f1(max(c['fltrust']['ddos_tm'] for c in pairs)), 'TM', f1(min(c['trimmed_mean_f1']['ddos_tm'] for c in pairs)), f1(max(c['trimmed_mean_f1']['ddos_tm'] for c in pairs)), '| pw FLT', f1(min(c['fltrust']['password_tm'] for c in pairs)), f1(max(c['fltrust']['password_tm'] for c in pairs)), 'TM', f1(min(c['trimmed_mean_f1']['password_tm'] for c in pairs)), f1(max(c['trimmed_mean_f1']['password_tm'] for c in pairs)))
print('time-matched gaps: ddos median', f1(st.median(gd2)), 'meanabs', f1(float(_n.mean(_n.abs(gd2)))), 'range', f1(min(gd2)), f1(max(gd2)), '| pw median', f1(st.median(gp2)), 'meanabs', f1(float(_n.mean(_n.abs(gp2)))), 'range', f1(min(gp2)), f1(max(gp2)), '| |pw|>|ddos| in', sum(abs(a) > abs(b) for a, b in zip(gp2, gd2)))
print('pairs', len(pairs), '| DDoS gap within 1pp', sum(abs(x) <= .01 for x in gd), 'range', f1(min(gd)), f1(max(gd)), '| password gap material', sum(abs(x) > .01 for x in gp), 'FLT>TM', sum(x > .01 for x in gp), 'range', f1(min(gp)), f1(max(gp)), 'median', f1(st.median(gp)))
