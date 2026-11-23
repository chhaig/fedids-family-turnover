#!/usr/bin/env python3
"""Third rule for the NF-ToN-IoT-v3 matched panel: FedAvg under clean / scaled_flip_a / scaled_flip_b,
seeds 29 and 43, five root draws (30 runs), using the archived runner, protocol and data on this node
unchanged (read-only). Outputs only under local-work/fedids_fedavg. One sequential background worker.
First call launches; later calls report status and copy finished step-600 scores to exchange/."""
import json, os, shutil, subprocess, time
from pathlib import Path
H = Path('local-work'); SRC = Path('external-workspace') / 'nf-root-seed-v1'
OUT = H / 'fedids_fedavg'; OUT.mkdir(parents=True, exist_ok=True)
EX = H / 'exchange' / 'fedids_fedavg'; EX.mkdir(parents=True, exist_ok=True)
rep = {'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
def find(name):
    hits = [p for p in SRC.rglob(name) if 'results' not in p.parts]
    return hits[0] if hits else None
runner = find('nf_trajectory_runner_v1.py'); proto = None
for p in SRC.rglob('*.json'):
    if 'results' in p.parts or p.stat().st_size > 200000: continue
    try:
        j = json.loads(p.read_text())
        if isinstance(j, dict) and j.get('id') == 'nf-root-seed-v1' and 'training' in j: proto = p; break
    except Exception: pass
py = H / 'research' / 'fedids-continuation-20261105' / '.venv' / 'bin' / 'python'
draws = sorted(d.parent for d in SRC.rglob('manifest.json') if d.parent.name == 'nf' and d.parent.parent.name.startswith('draw2026') and 'results' not in d.parts and (d.parent / 'train.npz').exists())
rep.update(runner=str(runner), protocol=str(proto), python=str(py), draws=[str(d) for d in draws])
rep['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
jobs = [(d, c, s) for d in draws for s in (29, 43) for c in ('clean', 'scaled_flip_a', 'scaled_flip_b')]
name = lambda d, c, s: f'nf-fedavg-{c}-{s}-{d.parent.name}'
marker = OUT / 'launched'
if runner and proto and py.exists() and len(draws) == 5 and not marker.exists():
    lines = ['#!/bin/bash', 'cd ' + str(runner.parent)]
    for d, c, s in jobs:
        o = OUT / name(d, c, s)
        lines.append(f'[ -f {o}/result.json ] && grep -q \'"completed": true\' {o}/result.json || {{ rm -rf {o}; nice -n 10 {py} {runner} --data {d} --protocol {proto} --output {o} --method fedavg --condition {c} --seed {s} > {o}.log 2>&1; }}')
    lines.append(f'echo ALLDONE > {OUT}/alldone')
    q = OUT / 'queue.sh'; q.write_text('\n'.join(lines) + '\n'); q.chmod(0o755)
    subprocess.Popen(['setsid', 'nohup', 'bash', str(q)], stdout=open(OUT / 'queue.out', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    marker.write_text(rep['time']); rep['launched'] = True
done = 0
for d, c, s in jobs:
    o = OUT / name(d, c, s); r = o / 'result.json'
    if r.exists():
        try: j = json.loads(r.read_text())
        except Exception: continue
        if j.get('completed'):
            done += 1; e = EX / o.name; e.mkdir(exist_ok=True)
            for f in ['result.json', 'cal-step600.npz', 'period_a-step600.npz', 'period_b-step600.npz']:
                if not (e / f).exists(): shutil.copy2(o / f, e / f)
        elif j.get('failure'): rep.setdefault('failures', {})[o.name] = j['failure']
rep.update(done=done, total=len(jobs), alldone=(OUT / 'alldone').exists())
logs = sorted(OUT.glob('*.log'), key=os.path.getmtime)
if logs: rep['last_log'] = logs[-1].name + ': ' + logs[-1].read_text()[-400:]
(EX / 'status.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
