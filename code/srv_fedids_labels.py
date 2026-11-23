#!/usr/bin/env python3
"""Read-only export of evaluation labels for the federated IDS matrices in a locally prepared training workspace.
Finds data directories (manifest.json + cal/period_a/period_b .npz) under external-workspace,
and writes, for each, the non-feature arrays (labels, weights, owners, families...) to
local-work/exchange/fedids/. Feature matrices are never copied. Nothing outside
local-work is written; no process is touched."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT = Path('external-workspace')
OUT = Path('local-work') / 'exchange' / 'fedids'
OUT.mkdir(parents=True, exist_ok=True)
ROLES = ['cal', 'period_a', 'period_b']
t0 = time.time()

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

found = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in ('results', 'artifacts', '__pycache__', '.git') and not d.startswith('step')]
    if 'manifest.json' in fn and all(r + '.npz' in fn for r in ROLES):
        found.append(Path(dp))
    if time.time() - t0 > 300:
        break

index, seen = [], {}
for d in found:
    entry = {'dir': str(d.relative_to(ROOT)), 'roles': {}}
    try:
        entry['manifest_keys'] = list(json.loads((d / 'manifest.json').read_text()).keys())
        man = json.loads((d / 'manifest.json').read_text())
        entry['owners'] = man.get('owners'); entry['weights'] = man.get('weights')
    except Exception as e:
        entry['manifest_error'] = str(e)
    for r in ROLES:
        p = d / (r + '.npz'); h = sha(p)
        info = {'sha256': h, 'bytes': p.stat().st_size}
        if h not in seen:
            keep = {}
            with np.load(p, allow_pickle=False) as z:
                info['keys'] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
                for k in z.files:
                    a = z[k]
                    if k == 'x' or a.ndim != 1:
                        continue
                    keep[k] = a
            np.savez_compressed(OUT / f'{h[:16]}.npz', **keep)
            seen[h] = f'{h[:16]}.npz'
        info['export'] = seen[h]
        entry['roles'][r] = info
    index.append(entry)
    if time.time() - t0 > 780:
        index.append({'truncated': True}); break

(OUT / 'index.json').write_text(json.dumps(index, indent=1))
print('data dirs', len(found), 'exported unique files', len(seen), 'seconds', round(time.time() - t0))
for e in index[:40]:
    print(e.get('dir'), {r: v.get('keys', {}).keys() and list(v.get('keys', {}).keys()) for r, v in e.get('roles', {}).items()} if 'roles' in e else e)
