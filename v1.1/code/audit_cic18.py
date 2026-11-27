"""Read-only audit of NF-CSE-CIC-IDS2018-v3: family x UTC hour, and per-host benign/attack per day.
No features are kept; only counts. Writes audit.json next to this script's output dir."""
import collections, json, sys, time, zipfile
import numpy as np, pandas as pd
ZIP, OUT = sys.argv[1], sys.argv[2]
t0 = time.monotonic()
fam_hour = collections.Counter(); host = collections.Counter(); nrows = 0
tmin, tmax = 1 << 62, 0
with zipfile.ZipFile(ZIP) as z:
    member = next(n for n in z.namelist() if n.endswith('/data/NF-CICIDS2018-v3.csv'))
    with z.open(member) as s:
        for f in pd.read_csv(s, usecols=['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack'],
                             dtype={'IPV4_SRC_ADDR': str, 'IPV4_DST_ADDR': str, 'Attack': str}, chunksize=500000, keep_default_na=False):
            nrows += len(f)
            st = f['FLOW_START_MILLISECONDS'].to_numpy()
            tmin = min(tmin, int(st.min())); tmax = max(tmax, int(f['FLOW_END_MILLISECONDS'].max()))
            hour = (st // 3600000).astype(np.int64)
            fam_hour.update(zip(f['Attack'].tolist(), hour.tolist()))
            day = (st // 86400000).astype(np.int64)
            lab = f['Label'].to_numpy().astype(int)
            for col in ('IPV4_DST_ADDR', 'IPV4_SRC_ADDR'):
                ip = f[col].to_numpy()
                internal = np.char.startswith(ip.astype('U15'), '172.31.')
                g = pd.DataFrame({'ip': ip[internal], 'day': day[internal], 'lab': lab[internal], 'fam': f['Attack'].to_numpy()[internal]})
                for (a, b, c, d), n in g.groupby(['ip', 'day', 'lab', 'fam']).size().items():
                    host[(col[5:8], a, int(b), int(c), d)] += int(n)
            if nrows % 5000000 == 0:
                print(json.dumps(dict(rows=nrows, s=round(time.monotonic() - t0))), flush=True)
out = dict(rows=nrows, start_min_ms=tmin, end_max_ms=tmax, seconds=time.monotonic() - t0,
           family_hour=[dict(family=a, hour=h, rows=n) for (a, h), n in sorted(fam_hour.items(), key=lambda kv: (kv[0][1], kv[0][0]))],
           host_day=[dict(side=s, ip=i, day=d, label=l, family=fm, rows=n) for (s, i, d, l, fm), n in host.items() if n >= 20])
json.dump(out, open(OUT, 'w'))
print(json.dumps(dict(done=True, rows=nrows, seconds=out['seconds'])), flush=True)
