#!/usr/bin/env python3
"""Generality extension on node 226, using an amended copy of the archived runner (rules_arch_runner_v2.py: adds
coord_median and krum_f1 and an optional architecture B; all else unchanged). 180 runs:
  NF-ToN-IoT-v3 matched cells (5 root draws x seeds 29, 43 x clean / scaled_flip_a / scaled_flip_b):
    coord_median, krum_f1 (arch A); fltrust, trimmed_mean_f1 (arch B)
  NF-CSE-CIC-IDS2018-v3 panel cells: coord_median, krum_f1 (arch A)
Archived data and protocols are read only; outputs only under FEDIDS_WORK_DIR/fedids_rules. Two sequential workers that
start after the CIC-IDS2018 panel workers have finished, each with the runner's 8% GPU cap and >=3 GiB free check."""
import json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
# Explicit local paths are required; this script never connects to a server.
D = Path(os.environ['FEDIDS_WORK_DIR'])
OUT = D / 'fedids_rules'; OUT.mkdir(parents=True, exist_ok=True)
EX = D / 'exchange' / 'fedids_rules'; EX.mkdir(parents=True, exist_ok=True)
C18 = D / 'fedids_cic18'; NFS = Path(os.environ['FEDIDS_NF_INPUT_DIR'])
py = Path(os.environ['FEDIDS_TRAIN_PYTHON'])
rep = {'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
inbox = D / 'exchange_in' / 'rules_arch_runner_v2.py'; runner = OUT / 'rules_arch_runner_v2.py'
if inbox.exists() and not runner.exists(): shutil.copy2(inbox, runner)
nfproto = None
for p in NFS.rglob('*.json'):
    if 'results' in p.parts or p.stat().st_size > 200000: continue
    try:
        j = json.loads(p.read_text())
        if isinstance(j, dict) and j.get('id') == 'nf-root-seed-v1' and 'training' in j: nfproto = p; break
    except Exception: pass
def extend(src, dst, note):
    if src and src.exists() and not dst.exists():
        p = json.loads(src.read_text()); p['methods'] = list(dict.fromkeys(p['methods'] + ['coord_median', 'krum_f1']))
        p['amendment'] = note; dst.write_text(json.dumps(p, indent=1))
    return dst
pnf = extend(nfproto, OUT / 'protocol-nf-rules.json', 'Copy of nf-root-seed-v1 with coord_median and krum_f1 added to methods; all other fields unchanged.')
pc18 = extend(C18 / 'protocol-seeds.json', OUT / 'protocol-cic18-rules.json', 'Copy of the CIC-IDS2018 panel protocol with coord_median and krum_f1 added to methods; all other fields unchanged.')
nfd = sorted(d.parent for d in NFS.rglob('manifest.json') if d.parent.name == 'nf' and d.parent.parent.name.startswith('draw2026') and 'results' not in d.parts and (d.parent / 'train.npz').exists())
c18d = sorted(d.parent for d in (C18 / 'data').glob('draw*/cic18/manifest.json'))
rep.update(runner=runner.exists(), nf_protocol=str(nfproto), nf_draws=len(nfd), cic18_draws=len(c18d))
jobs = []
for d in nfd:
    for s in (29, 43):
        for c in ('clean', 'scaled_flip_a', 'scaled_flip_b'):
            for m, a in (('coord_median', 'A'), ('krum_f1', 'A'), ('fltrust', 'B'), ('trimmed_mean_f1', 'B')):
                jobs.append(('nf', d, pnf, m, c, s, a, f'nf-{m}-{c}-{s}-{d.parent.name}-arch{a}'))
for d in c18d:
    for s in (29, 43):
        for c in ('clean', 'scaled_flip_a', 'scaled_flip_b'):
            for m in ('coord_median', 'krum_f1'):
                jobs.append(('cic18', d, pc18, m, c, s, 'A', f'cic18-{m}-{c}-{s}-{d.parent.name}-archA'))
SCORE = C18 / 'score_seen.py'
marker = OUT / 'launched'
if runner.exists() and py.exists() and len(nfd) == 5 and len(c18d) == 5 and not marker.exists():
    for w in range(2):
        lines = ['#!/bin/bash', f'while [ ! -f {C18}/worker0.done ] || [ ! -f {C18}/worker1.done ]; do sleep 60; done', 'cd ' + str(OUT)]
        for i, (ds, d, p, m, c, s, a, n) in enumerate(jobs):
            if i % 2 != w: continue
            o = OUT / n; fin = f'{o}/seen-step600.npz' if ds == 'cic18' else f'{o}/period_b-step600.npz'
            post = f' && {py} {SCORE} {o} {d} >> {o}.log 2>&1' if ds == 'cic18' else ''
            lines.append(f'[ -f {fin} ] || {{ rm -rf {o}; {py} {runner} --data {d} --protocol {p} --output {o} --method {m} --condition {c} --seed {s} --arch {a} > {o}.log 2>&1{post}; }}')
        lines.append(f'echo DONE > {OUT}/worker{w}.done')
        q = OUT / f'queue{w}.sh'; q.write_text('\n'.join(lines) + '\n'); q.chmod(0o755)
        subprocess.Popen(['setsid', 'nohup', 'bash', str(q)], stdout=open(OUT / f'queue{w}.out', 'w'), stderr=subprocess.STDOUT, start_new_session=True)
    marker.write_text(rep['time']); rep['launched'] = True
done = 0
for ds, d, p, m, c, s, a, n in jobs:
    o = OUT / n; e = EX / n
    roles = ['cal', 'period_a', 'period_b'] + (['seen'] if ds == 'cic18' else [])
    if all((o / f'{r}-step600.npz').exists() for r in roles) and (o / 'result.json').exists():
        try: j = json.loads((o / 'result.json').read_text())
        except Exception: continue
        if not j.get('completed'): continue
        done += 1
        if not (e / 'scores.npz').exists():
            e.mkdir(exist_ok=True)
            np.savez_compressed(e / 'scores.npz', **{r: np.load(o / f'{r}-step600.npz')['score'].astype(np.float32) for r in roles})
            keep = {k: j[k] for k in j if k not in ('rounds', 'probes')}; keep['last_round'] = j['rounds'][-1] if j.get('rounds') else None
            (e / 'result.json').write_text(json.dumps(keep, indent=1))
    elif (o / 'result.json').exists():
        try:
            j = json.loads((o / 'result.json').read_text())
            if j.get('failure'): rep.setdefault('failures', {})[n] = j['failure']
        except Exception: pass
rep.update(done=done, total=len(jobs), workers_done=[p.name for p in OUT.glob('worker*.done')])
c18 = json.loads((D / 'exchange' / 'fedids_cic18' / 'status.json').read_text()) if (D / 'exchange' / 'fedids_cic18' / 'status.json').exists() else {}
rep['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
logs = sorted(OUT.glob('*.log'), key=os.path.getmtime)
if logs: rep['last_log'] = logs[-1].name + ': ' + logs[-1].read_text()[-300:]
(EX / 'status.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
