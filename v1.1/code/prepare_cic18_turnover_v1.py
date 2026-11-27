"""NF-CSE-CIC-IDS2018-v3 arrays for the family-turnover panel, following the frozen protocol.
Stages (each fits in one short call): select (time columns only), extract (raw lines of selected rows), build (features).
Labels are never used for eligibility, client assignment, sampling or root selection."""
import collections, datetime, hashlib, json, sys, time, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / 'protocols' / 'cic18_turnover_v1.protocol.json'
P = json.loads(PROTOCOL.read_text())
MEMBER_SUFFIX = '/data/NF-CICIDS2018-v3.csv'
NROWS = 20115529

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def splitmix(row, seed):
    z = row.astype(np.uint64) ^ np.uint64(seed)
    with np.errstate(over='ignore'):
        z = z + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return z ^ (z >> np.uint64(31))

ms = lambda s: int(datetime.datetime.fromisoformat(s).timestamp() * 1000)

def select(zip_path, out):
    import pandas as pd
    T = {k: (ms(v['start_inclusive']), ms(v['end_exclusive'])) for k, v in P['time_utc'].items()}
    caps = P['sampling']['caps']; K = P['clients']['count']
    pools = {}; eligible = collections.Counter(); seenpool = {}; off = 0; t0 = time.monotonic()
    def keep(key, rows, pri, cap, store):
        if key in store:
            rows = np.concatenate([store[key][0], rows]); pri = np.concatenate([store[key][1], pri])
        if len(rows) > cap:
            at = np.argpartition(pri, cap - 1)[:cap]; rows, pri = rows[at], pri[at]
        store[key] = (rows, pri)
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(MEMBER_SUFFIX))
        with z.open(member) as s:
            for f in pd.read_csv(s, usecols=['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'], chunksize=1000000):
                st = f['FLOW_START_MILLISECONDS'].to_numpy(); en = f['FLOW_END_MILLISECONDS'].to_numpy()
                row = np.arange(off, off + len(f), dtype=np.int64); off += len(f)
                cl = (splitmix(row, 20261127) % np.uint64(K)).astype(np.int64)
                pri = splitmix(row, 20261126); pri_seen = splitmix(row, 20261129)
                role = {
                    'train': (st >= T['train'][0]) & (en < T['train'][1]),
                    'cal': (st >= T['cal'][0]) & (en < T['cal'][1]),
                    'period_a': (st >= T['period_a'][0]) & (en < T['period_a'][1]),
                    'period_b': (st >= T['period_b'][0]) & (en < T['period_b'][1]),
                }
                day = st // 86400000
                for r, m in role.items():
                    for c in range(K):
                        mm = m & (cl == c)
                        if not mm.any(): continue
                        days = np.unique(day[mm]) if r == 'period_a' else [-1]
                        for d in days:
                            m2 = mm & (day == d) if r == 'period_a' else mm
                            key = (r, c, int(d)); eligible[key] += int(m2.sum())
                            keep(key, row[m2], pri[m2], caps[r], pools)
                            if r == 'train':  # candidate pool for the held-out seen set (excludes train picks later)
                                keep(('seen', c, -1), row[m2], pri_seen[m2], caps['seen'] + caps['train'], seenpool)
    assert off == NROWS, off
    sel = {}
    train_ids = set()
    for key, (rows, _) in pools.items():
        sel[key] = np.sort(rows)
        if key[0] == 'train': train_ids.update(rows.tolist())
    for (r, c, d), (rows, pri) in seenpool.items():
        ok = ~np.isin(rows, np.fromiter(train_ids, np.int64))
        rows, pri = rows[ok], pri[ok]
        at = np.argsort(pri)[:caps['seen']]; sel[('seen', c, -1)] = np.sort(rows[at])
        eligible[('seen', c, -1)] = eligible[('train', c, -1)] - len(sel[('train', c, -1)])
    cells = [dict(role=r, client=c, day=d, eligible=eligible[(r, c, d)], sampled=int(len(v))) for (r, c, d), v in sorted(sel.items())]
    allrows = np.concatenate(list(sel.values())); assert len(allrows) == len(np.unique(allrows))
    np.savez_compressed(out, **{f'{r}|{c}|{d}': v for (r, c, d), v in sel.items()})
    json.dump(dict(cells=cells, seconds=time.monotonic() - t0, rows=off), open(str(out) + '.json', 'w'), indent=1)
    print(json.dumps(dict(selected=int(len(allrows)), seconds=round(time.monotonic() - t0))))

def extract(zip_path, sel_path, out):
    z0 = np.load(sel_path); tag = np.zeros(NROWS, dtype=np.int16) - 1
    names = sorted(z0.files)
    for i, k in enumerate(names): tag[z0[k]] = i
    t0 = time.monotonic(); n = 0
    with zipfile.ZipFile(zip_path) as z, open(out, 'w') as w:
        member = next(n for n in z.namelist() if n.endswith(MEMBER_SUFFIX))
        with z.open(member) as s:
            header = s.readline().decode().rstrip('\n'); w.write('_row,_cell,' + header + '\n')
            for i, line in enumerate(s):
                t = tag[i]
                if t >= 0:
                    w.write(f'{i},{names[t]},' + line.decode()); n += 1
    print(json.dumps(dict(extracted=n, seconds=round(time.monotonic() - t0))))

def build(raw_csv, sel_json, outdir):
    import pandas as pd
    outdir = Path(outdir); outdir.mkdir(exist_ok=False)
    df = pd.read_csv(raw_csv, dtype={'IPV4_SRC_ADDR': str, 'IPV4_DST_ADDR': str, 'Attack': str, '_cell': str}, keep_default_na=False)
    meta = json.load(open(sel_json)); w = {f"{c['role']}|{c['client']}|{c['day']}": c['eligible'] / c['sampled'] for c in meta['cells']}
    df['_role'] = df['_cell'].str.split('|').str[0]; df['_client'] = df['_cell'].str.split('|').str[1].astype(int)
    df['_weight'] = df['_cell'].map(w)
    drop = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'DNS_QUERY_ID', 'Label', 'Attack']
    cats = ['PROTOCOL', 'L7_PROTO']
    cols = [c for c in df.columns if not c.startswith('_')]
    numeric = [c for c in cols if c not in drop + cats]
    tr = df[df['_role'] == 'train'].sort_values(['_client', 'FLOW_END_MILLISECONDS', '_row']).reset_index(drop=True)
    raw = tr[numeric].to_numpy(np.float64); assert np.isfinite(raw).all() and (raw >= 0).all()
    lg = np.log1p(raw); mean, std = lg.mean(0), lg.std(0); std[std == 0] = 1
    vocab = {c: sorted(int(v) for v in tr[c].unique()) for c in cats}
    feats = numeric + [f'{c}={v}' for c in cats for v in vocab[c] + ['UNKNOWN']]
    json.dump(dict(numeric=numeric, mean=mean.tolist(), std=std.tolist(), categorical_vocabulary=vocab, model_features=feats, fit_rows=len(tr), fit_uses_labels=False), open(outdir / 'preprocessing.json', 'w'), indent=1)
    names = P['clients']['names']
    def arrays(fr):
        r = fr[numeric].to_numpy(np.float64); assert np.isfinite(r).all() and (r >= 0).all()
        parts = [np.clip((np.log1p(r) - mean) / std, -8, 8).astype(np.float32)]
        for c, vs in vocab.items():
            x = fr[c].to_numpy(); parts.append(np.column_stack([(x == v).astype(np.float32) for v in vs] + [(~np.isin(x, vs)).astype(np.float32)]))
        x = np.concatenate(parts, 1); x[x == 0] = 0
        host = np.where(fr['IPV4_DST_ADDR'].str.startswith('172.31.'), fr['IPV4_DST_ADDR'], np.where(fr['IPV4_SRC_ADDR'].str.startswith('172.31.'), fr['IPV4_SRC_ADDR'], 'external'))
        return dict(x=x, y=fr['Label'].to_numpy(np.uint8), owner=np.array([names[i] for i in fr['_client']], dtype='U16'),
                    family=fr['Attack'].to_numpy(dtype='U32'), weight=fr['_weight'].to_numpy(np.float64),
                    host=host.astype('U16'), day=pd.to_datetime(fr['FLOW_START_MILLISECONDS'], unit='ms', utc=True).dt.strftime('%Y-%m-%d').to_numpy(dtype='U21'),
                    time=fr['FLOW_START_MILLISECONDS'].to_numpy(np.int64), original_index=fr['_row'].to_numpy(np.int64),
                    model_input_hash=np.array([hashlib.sha256(v.astype('<f4').tobytes()).hexdigest() for v in x], dtype='U64'))
    out = {}
    for role in ['train', 'cal', 'period_a', 'period_b', 'seen']:
        fr = tr if role == 'train' else df[df['_role'] == role].sort_values(['_client', 'FLOW_END_MILLISECONDS', '_row']).reset_index(drop=True)
        out[role] = arrays(fr); np.savez_compressed(outdir / f'{role}.npz', **out[role])
    a = out['train']
    elig = {c['client']: c['eligible'] for c in meta['cells'] if c['role'] == 'train'}
    pop = sum(elig.values()) - 64 * len(names)
    weights = [(elig[i] - 64) / pop for i in range(len(names))]
    draws = {}
    for s in P['root']['draw_seeds']:
        rng = np.random.default_rng(s); root = []
        for o in names:
            ids = np.flatnonzero(a['owner'] == o); root.extend(np.sort(rng.choice(ids, 64, replace=False)).tolist())
        root = np.array(root, np.int64); private = np.setdiff1d(np.arange(len(a['y'])), root)
        f = outdir / f'partition-draw{s}.npz'; np.savez_compressed(f, root=root, private=private)
        draws[s] = dict(file=f.name, root_positives=int(a['y'][root].sum()))
    fam = lambda o: {k: int(v) for k, v in collections.Counter(o['family'].tolist()).items()}
    man = dict(dataset='cic18', scope='NF-CSE-CIC-IDS2018-v3 family-turnover panel v1', owners=names, weights=weights,
               attackers=P['clients']['attackers'], rng_base=20264400, input_dim=len(feats), root_rows=384,
               protocol_sha256=digest(PROTOCOL), preparer_sha256=digest(Path(__file__)),
               files={f.name: digest(f) for f in sorted(outdir.glob('*.npz'))}, root_draws=draws,
               splits={r: dict(rows=len(o['y']), positives=int(o['y'].sum()), families=fam(o)) for r, o in out.items()}, cells=meta['cells'])
    json.dump(man, open(outdir / 'manifest-base.json', 'w'), indent=1)
    print(json.dumps(dict(input_dim=len(feats), splits=man['splits'], weights=[round(x, 4) for x in weights], draws=draws)))

if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'select': select(sys.argv[2], sys.argv[3])
    elif st == 'extract': extract(sys.argv[2], sys.argv[3], sys.argv[4])
    elif st == 'build': build(sys.argv[2], sys.argv[3], sys.argv[4])
