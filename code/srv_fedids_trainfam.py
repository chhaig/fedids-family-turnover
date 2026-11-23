#!/usr/bin/env python3
"""Read-only: per data directory, count training rows by label/family/owner (no features)."""
import json, os, collections
from pathlib import Path
import numpy as np
ROOT = Path('external-workspace')
OUT = Path('local-work') / 'exchange' / 'fedids'; OUT.mkdir(parents=True, exist_ok=True)
want = ['cross-testbed-adaptive-v1/data/nf', 'cross-testbed-adaptive-v1/data/cic', 'cross-testbed-adaptive-v1/data/iot', 'nf-unsw-study-matrix-v1/data/draw-17']
res = {}
for w in want:
    d = ROOT / w; r = {}
    try:
        with np.load(d / 'train.npz', allow_pickle=False) as z:
            y = z['y']; fam = z['family'] if 'family' in z.files else None
            wt = z['weight'] if 'weight' in z.files else np.ones(len(y))
            r['rows'] = int(len(y)); r['attack_rows'] = int((y == 1).sum())
            if fam is not None:
                r['families'] = {str(k): int(((y == 1) & (fam == k)).sum()) for k in np.unique(fam[y == 1])}
            for k in ('day',):
                if k in z.files: r['days'] = sorted(map(str, np.unique(z[k])))[:40]
            r['keys'] = list(z.files)
        if (d / 'partition.npz').exists():
            with np.load(d / 'partition.npz') as p:
                root = p['root']; r['root_families'] = {str(k): int(((y[root] == 1) & (fam[root] == k)).sum()) for k in np.unique(fam[root][y[root] == 1])} if fam is not None else None
                r['root_benign'] = int((y[root] == 0).sum())
    except Exception as e:
        r['error'] = repr(e)
    res[w] = r
(OUT / 'train_families.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1)[:4000])
