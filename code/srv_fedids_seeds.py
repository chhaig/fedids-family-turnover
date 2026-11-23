#!/usr/bin/env python3
"""Additional optimization seeds (29, 43) for FLTrust and trimmed mean on CIC-IDS2017 and IoT-23,
using the existing frozen cross-testbed runner and data on this node (read-only). Outputs go only to
local-work/fedids_seeds. One background worker, sequential jobs, the runner's own 8% GPU
allocator cap and its >=3 GiB free-memory check. No other process is touched.
First call launches; later calls report status and copy finished step-600 scores to exchange/."""
import json, os, shutil, subprocess, sys, glob, time
from pathlib import Path
H = Path('local-work'); SRC = Path('external-workspace') / 'cross-testbed-adaptive-v1'
OUT = H / 'fedids_seeds'; OUT.mkdir(parents=True, exist_ok=True)
EX = H / 'exchange' / 'fedids_seeds'; EX.mkdir(parents=True, exist_ok=True)
rep = {'time': time.strftime('%Y-%m-%dT%H:%M:%S')}

def find(name):
    hits = [p for p in SRC.rglob(name) if 'results' not in p.parts]
    return hits[0] if hits else None
runner = find('cross_testbed_runner_v1.py'); proto = find('cross-testbed-adaptive-v1.json')
rep['runner'] = str(runner); rep['protocol'] = str(proto)
pys = [Path(sys.executable)]
py = None
for c in pys:
    if c.exists():
        r = subprocess.run([str(c), '-c', 'import torch,numpy;print(torch.__version__, torch.cuda.is_available())'], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and 'True' in r.stdout:
            py = c; rep['python'] = str(c); rep['torch'] = r.stdout.strip(); break
rep['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()

fix = OUT / 'python_fix_v2'
if not fix.exists():
    # stop only this script's own queue and runner processes (launched with the wrong interpreter), then relaunch
    me = subprocess.run(['pgrep', '-f', str(OUT)], capture_output=True, text=True).stdout.split()
    for pid in me:
        if int(pid) != os.getpid():
            subprocess.run(['kill', pid])
    time.sleep(3)
    for d in OUT.iterdir():
        if d.is_dir() or d.suffix == '.log' or d.name in ('launched', 'queue.sh', 'queue.out', 'alldone'):
            shutil.rmtree(d) if d.is_dir() else d.unlink()
    fix.write_text(rep['time']); rep['relaunched_with_fixed_python'] = me
proto2 = OUT / 'protocol-seeds.json'
if proto and not proto2.exists():
    p = json.loads(proto.read_text()); p['training']['seeds'] = [17, 29, 43]
    p['amendment'] = 'Copy of cross-testbed-adaptive-v1 with seeds 29 and 43 added for a seed-replication of FLTrust and trimmed mean; all other fields unchanged.'
    proto2.write_text(json.dumps(p, indent=1))
jobs = [(ds, m, c, s) for ds in ('cic', 'iot') for s in (29, 43) for m in ('fltrust', 'trimmed_mean_f1') for c in ('clean', 'scaled_flip_a', 'scaled_flip_b')]
data = {ds: next((d for d in SRC.rglob('manifest.json') if d.parent.name == ds and 'results' not in d.parts), None) for ds in ('cic', 'iot')}
rep['data'] = {k: str(v.parent) if v else None for k, v in data.items()}
queue = OUT / 'queue.sh'; marker = OUT / 'launched'
if runner and py and all(data.values()) and not marker.exists():
    lines = ['#!/bin/bash', 'cd ' + str(runner.parent)]
    for ds, m, c, s in jobs:
        o = OUT / f'{ds}-{m}-{c}-{s}'
        lines.append(f'[ -f {o}/result.json ] && grep -q \'"completed": true\' {o}/result.json || {{ rm -rf {o}; {py} {runner} --data {data[ds].parent} --protocol {proto2} --output {o} --method {m} --condition {c} --seed {s} > {o}.log 2>&1; }}')
    lines.append(f'echo ALLDONE > {OUT}/alldone')
    queue.write_text('\n'.join(lines) + '\n'); queue.chmod(0o755)
    subprocess.Popen(['setsid', 'nohup', 'bash', str(queue)], stdout=open(OUT / 'queue.out', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    marker.write_text(rep['time']); rep['launched'] = True
done = []
for ds, m, c, s in jobs:
    o = OUT / f'{ds}-{m}-{c}-{s}'; r = o / 'result.json'
    if r.exists():
        try:
            j = json.loads(r.read_text())
        except Exception:
            continue
        if j.get('completed'):
            done.append(o.name); e = EX / o.name; e.mkdir(exist_ok=True)
            for f in ['result.json', 'cal-step600.npz', 'period_a-step600.npz', 'period_b-step600.npz']:
                if not (e / f).exists(): shutil.copy2(o / f, e / f)
        elif j.get('failure'):
            rep.setdefault('failures', {})[o.name] = j['failure']
rep['done'] = len(done); rep['total'] = len(jobs); rep['alldone'] = (OUT / 'alldone').exists()
logs = sorted(OUT.glob('*.log'), key=os.path.getmtime)
if logs: rep['last_log'] = logs[-1].name + ': ' + logs[-1].read_text()[-600:]
(EX / 'status.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
