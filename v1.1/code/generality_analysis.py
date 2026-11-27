"""Generality analyses for the family-turnover paper (second benchmark, more rules, second architecture).
Same definitions as the original analysis scripts: weighted 99th-percentile benign threshold (1% FPR),
oracle recall, weighted AUC with ties counted 1/2, materiality 1 pp, paired flow-level bootstrap, pool exchange,
family decomposition, flat-Dirichlet mix-stability profile, seen-family control, historical-threshold FPR.
Usage: python3 generality_analysis.py cic18|nf [bootstrap <draw>]"""
import json, os, sys, glob, collections, itertools
from pathlib import Path
import numpy as np
PACKAGE = Path(__file__).resolve().parents[1]
REPRO = Path(os.environ.get('FEDIDS_V1_ROOT', PACKAGE.parent))
CIC18_LABELS = PACKAGE / 'labels' / 'cic18'
CIC18_SCORES = PACKAGE / 'scores' / 'cic18-panel'
EXTENSION_SCORES = PACKAGE / 'scores' / 'rules-arch-extension'
OUT = Path(os.environ.get('FEDIDS_ANALYSIS_OUT', PACKAGE / 'outputs'))
MAT = 0.01

def thr(s, w, q=.99):
    o = np.argsort(s, kind='stable'); c = np.cumsum(w[o]); return s[o][min(np.searchsorted(c, q * c[-1]), len(s) - 1)]
def recall(s, w, t): return float((w * (s > t)).sum() / w.sum()) if len(s) else float('nan')
def auc(sb, wb, sa, wa):
    s = np.concatenate([sb, sa]); w = np.concatenate([wb, wa]); a = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
    o = np.argsort(s, kind='stable'); s, w, a = s[o], w[o], a[o]
    u, start = np.unique(s, return_index=True); end = np.r_[start[1:], len(s)]
    wbg = np.add.reduceat(w * (1 - a), start); wag = np.add.reduceat(w * a, start)
    below = np.r_[0, np.cumsum(wbg)[:-1]]
    return float(((below + .5 * wbg) * wag).sum() / (wb.sum() * wa.sum()))

def load_cic18_labels():
    L = {}
    for r in ['cal', 'period_a', 'period_b', 'seen']:
        z = np.load(CIC18_LABELS / f'{r}.npz'); L[r] = dict(y=z['y'].astype(int), w=z['weight'].astype(float), f=z['family'].astype(str))
    return L

def runs_cic18():
    R = {}
    for d in glob.glob(str(CIC18_SCORES / 'cic18-*')) + glob.glob(str(EXTENSION_SCORES / 'cic18-*')):
        n = os.path.basename(d)
        if not os.path.exists(d + '/scores.npz'): continue
        p = n.split('-'); m, c, s, draw = p[1], p[2], int(p[3]), p[4]
        z = np.load(d + '/scores.npz'); R[(c, s, draw, m, 'A')] = {k: z[k].astype(float) for k in z.files}
    return R

def labels_by_hash(h):
    z = np.load(REPRO / 'labels' / (h[:16] + '.npz'))
    return dict(y=z['y'].astype(int), w=z['weight'].astype(float), f=z['family'].astype(str))

def runs_nf():
    R = {}; L = {}
    def add(d, m, c, s, draw, arch, slim=True):
        res = json.load(open(d + '/result.json'))
        if os.path.exists(d + '/scores.npz'):
            z = np.load(d + '/scores.npz'); sc = {k: z[k].astype(float) for k in z.files}
        else:
            sc = {r: np.load(f'{d}/{r}-step600.npz')['score'].astype(float) for r in ['cal', 'period_a', 'period_b']}
        for r in ['cal', 'period_a', 'period_b']:
            h = res['input_sha256'][r + '.npz']
            if r not in L: L[r] = labels_by_hash(h); L[r + '_hash'] = h
            assert L[r + '_hash'] == h, 'evaluation set differs across draws'
        R[(c, s, draw, m, arch)] = sc
    for d in glob.glob(str(REPRO / 'scores/nf-root-seed-v1-matrix/nf-*')) + glob.glob(str(REPRO / 'scores/fedavg-matched-new/nf-*')):
        p = os.path.basename(d).split('-'); add(d, p[1], p[2], int(p[3]), p[4], 'A')
    for d in glob.glob(str(EXTENSION_SCORES / 'nf-*')):
        if not os.path.exists(d + '/scores.npz'): continue
        p = os.path.basename(d).split('-'); add(d, p[1], p[2], int(p[3]), p[4], p[5][-1])
    return R, L

def model_stats(sc, L, q=.99, fams=None):
    out = {}
    tc = thr(sc['cal'][L['cal']['y'] == 0], L['cal']['w'][L['cal']['y'] == 0])
    for r in [k for k in ('period_a', 'period_b', 'seen') if k in sc]:
        y, w, f = L[r]['y'], L[r]['w'], L[r]['f']; s = sc[r]; b = y == 0; a = y == 1
        t = thr(s[b], w[b], q)
        fam = {fm: recall(s[a & (f == fm)], w[a & (f == fm)], t) for fm in sorted(set(f[a]))}
        out[r] = dict(oracle_recall=recall(s[a], w[a], t), auc=auc(s[b], w[b], s[a], w[a]) if q == .99 else None,
                      hist_fpr=float((w[b] * (s[b] > tc)).sum() / w[b].sum()), family_recall=fam, thr=float(t))
    return out

def pool_exchange(s1, s2, L):
    g = {}
    for bp in ('period_a', 'period_b'):
        for ap in ('period_a', 'period_b'):
            r = []
            for s in (s1, s2):
                yb = L[bp]['y'] == 0; t = thr(s[bp][yb], L[bp]['w'][yb])
                ya = L[ap]['y'] == 1; r.append(recall(s[ap][ya], L[ap]['w'][ya], t))
            g[(bp, ap)] = r[0] - r[1]
    sgn = {k: (0 if abs(v) <= MAT else (1 if v > 0 else -1)) for k, v in g.items()}
    allm = all(v != 0 for v in sgn.values())
    attack = allm and sgn[('period_a', 'period_a')] == sgn[('period_b', 'period_a')] and sgn[('period_a', 'period_b')] == sgn[('period_b', 'period_b')] and sgn[('period_a', 'period_a')] != sgn[('period_a', 'period_b')]
    benign = allm and sgn[('period_a', 'period_a')] == sgn[('period_a', 'period_b')] and sgn[('period_b', 'period_a')] == sgn[('period_b', 'period_b')] and sgn[('period_a', 'period_a')] != sgn[('period_b', 'period_a')]
    return dict(gaps={f'{k[0]}|{k[1]}': v for k, v in g.items()}, attack_driven=bool(attack), benign_driven=bool(benign))

def decomposition(s1, s2, L, early_fams, late_fam, rng):
    res = {}
    for bp in ('period_a', 'period_b'):
        fr = []
        for s in (s1, s2):
            yb = L[bp]['y'] == 0; t = thr(s[bp][yb], L[bp]['w'][yb]); d = {}
            for p in ('period_a', 'period_b'):
                y, w, f = L[p]['y'], L[p]['w'], L[p]['f']
                for fm in set(f[y == 1]):
                    m = (y == 1) & (f == fm); d[fm] = recall(s[p][m], w[m], t)
            fr.append(d)
        dfam = {fm: fr[0][fm] - fr[1][fm] for fm in fr[0]}
        y, w, f = L['period_a']['y'], L['period_a']['w'], L['period_a']['f']
        wts = {fm: w[(y == 1) & (f == fm)].sum() for fm in early_fams}
        DE = sum(wts[fm] * dfam[fm] for fm in early_fams) / sum(wts.values()); dL = dfam[late_fam]
        star = DE / (DE - dL) if ((DE > 0) != (dL > 0)) and abs(dL) > MAT and abs(DE) > 0 else None
        fams = early_fams + [late_fam]; v = np.array([dfam[fm] for fm in fams])
        pis = rng.dirichlet(np.ones(len(fams)), 20000); D = pis @ v
        prof = dict(first=float((D > MAT).mean()), second=float((D < -MAT).mean()), tie=float((np.abs(D) <= MAT).mean()))
        res[bp] = dict(d_family=dfam, D_earlier_mix=DE, d_later=dL, crossover_later_share=star, profile=prof,
                       contested=bool(prof['first'] >= .1 and prof['second'] >= .1))
    return res

def bootstrap_pair(s1, s2, L, reps, rng):
    out = {}
    for p in ('period_a', 'period_b'):
        y, w = L[p]['y'], L[p]['w']; ib = np.flatnonzero(y == 0); ia = np.flatnonzero(y == 1); diffs = []
        for _ in range(reps):
            b = ib[rng.integers(0, len(ib), len(ib))]; a = ia[rng.integers(0, len(ia), len(ia))]
            r = [recall(s[p][a], w[a], thr(s[p][b], w[b])) for s in (s1, s2)]; diffs.append(r[0] - r[1])
        out[p] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    return out

def summarise(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None]); return dict(n=len(v), median=float(np.median(v)) if len(v) else None, min=float(v.min()) if len(v) else None, max=float(v.max()) if len(v) else None)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = sys.argv[1]; rng = np.random.default_rng(20261127)
    if ds == 'cic18':
        L = load_cic18_labels(); R = runs_cic18(); early = ['DDoS_attacks-LOIC-HTTP', 'DDOS_attack-HOIC', 'DDOS_attack-LOIC-UDP']; late = 'Bot'
    else:
        R, L = runs_nf(); early = ['password', 'xss']; late = 'Backdoor'
    cells = sorted({k[:3] for k in R}); rules = sorted({(k[3], k[4]) for k in R})
    stats = {k: model_stats(v, L) for k, v in R.items()}
    report = dict(dataset=ds, models=len(R), cells=len(cells), rules=[f'{m}/{a}' for m, a in rules])
    if len(sys.argv) > 2 and sys.argv[2] == 'bootstrap':
        draw = sys.argv[3]; bs = {}
        for c in cells:
            if c[2] != draw: continue
            k1, k2 = c + ('fltrust', 'A'), c + ('trimmed_mean_f1', 'A')
            if k1 in R and k2 in R: bs['|'.join(map(str, c))] = bootstrap_pair(R[k1], R[k2], L, 400, rng)
        json.dump(bs, open(OUT / f'{ds}-bootstrap-{draw}.json', 'w'), indent=1); print(json.dumps(dict(pairs=len(bs)))); return
    # per-model summary
    report['per_rule'] = {}
    for m, a in rules:
        ks = [k for k in R if k[3] == m and k[4] == a]
        report['per_rule'][f'{m}/{a}'] = {p: dict(oracle=summarise([stats[k][p] for k in ks], 'oracle_recall'), auc=summarise([stats[k][p] for k in ks], 'auc'), hist_fpr=summarise([stats[k][p] for k in ks], 'hist_fpr')) for p in stats[ks[0]]}
    # matched FLTrust vs trimmed mean (each architecture)
    def matched(m1, m2, arch):
        rows = []
        for c in cells:
            k1, k2 = c + (m1, arch), c + (m2, arch)
            if k1 not in R or k2 not in R: continue
            row = dict(cell='|'.join(map(str, c)))
            for q in (.999, .995, .99, .98, .95):
                st1, st2 = (stats[k1], stats[k2]) if q == .99 else (model_stats(R[k1], L, q), model_stats(R[k2], L, q))
                dE = st1['period_a']['oracle_recall'] - st2['period_a']['oracle_recall']; dL = st1['period_b']['oracle_recall'] - st2['period_b']['oracle_recall']
                row[f'q{q}'] = dict(dE=dE, dL=dL, reversal=bool(abs(dE) > MAT and abs(dL) > MAT and (dE > 0) != (dL > 0)))
            row['auc'] = dict(dE=stats[k1]['period_a']['auc'] - stats[k2]['period_a']['auc'], dL=stats[k1]['period_b']['auc'] - stats[k2]['period_b']['auc'])
            row['auc']['reversal'] = bool((row['auc']['dE'] > 0) != (row['auc']['dL'] > 0))
            if 'seen' in stats[k1]:
                row['seen_gap'] = stats[k1]['seen']['oracle_recall'] - stats[k2]['seen']['oracle_recall']
                row['seen_recall'] = [stats[k1]['seen']['oracle_recall'], stats[k2]['seen']['oracle_recall']]
            if ds == 'nf':  # calibration-evening DDoS (seen family) at the calibration benign threshold
                y, w = L['cal']['y'], L['cal']['w']; g = []
                for k in (k1, k2):
                    s = R[k]['cal']; t = thr(s[y == 0], w[y == 0]); g.append(recall(s[y == 1], w[y == 1], t))
                row['seen_gap'] = g[0] - g[1]; row['seen_recall'] = g
            row['pool'] = pool_exchange(R[k1], R[k2], L)
            row['decomp'] = decomposition(R[k1], R[k2], L, early, late, rng)
            rows.append(row)
        if not rows: return None
        q = rows[0]
        agg = dict(pairs=len(rows),
                   reversal={f'{x}': sum(r[f'q{x}']['reversal'] for r in rows) for x in (.999, .995, .99, .98, .95)},
                   first_leads_earlier=sum(r['q0.99']['dE'] > MAT for r in rows), second_leads_earlier=sum(r['q0.99']['dE'] < -MAT for r in rows),
                   first_leads_later=sum(r['q0.99']['dL'] > MAT for r in rows), second_leads_later=sum(r['q0.99']['dL'] < -MAT for r in rows),
                   median_dE=float(np.median([r['q0.99']['dE'] for r in rows])), median_dL=float(np.median([r['q0.99']['dL'] for r in rows])),
                   auc_reversal=sum(r['auc']['reversal'] for r in rows),
                   attack_driven=sum(r['pool']['attack_driven'] for r in rows), benign_driven=sum(r['pool']['benign_driven'] for r in rows),
                   seen_gap_max_abs=float(max(abs(r['seen_gap']) for r in rows)) if 'seen_gap' in q else None,
                   seen_recall_range=[float(min(min(r['seen_recall']) for r in rows)), float(max(max(r['seen_recall']) for r in rows))] if 'seen_recall' in q else None)
        for bp in ('period_a', 'period_b'):
            st = [r['decomp'][bp]['crossover_later_share'] for r in rows if r['decomp'][bp]['crossover_later_share'] is not None]
            opp = sum(1 for r in rows if (r['decomp'][bp]['D_earlier_mix'] > MAT and r['decomp'][bp]['d_later'] < -MAT) or (r['decomp'][bp]['D_earlier_mix'] < -MAT and r['decomp'][bp]['d_later'] > MAT))
            prof = [r['decomp'][bp]['profile'] for r in rows]
            agg[f'decomp_{bp}'] = dict(opposite_material=opp, crossovers=len(st), crossover_median=float(np.median(st)) if st else None,
                                       crossover_range=[float(min(st)), float(max(st))] if st else None,
                                       contested=sum(r['decomp'][bp]['contested'] for r in rows),
                                       profile_median={k: float(np.median([p[k] for p in prof])) for k in ('first', 'second', 'tie')})
        return dict(aggregate=agg, rows=rows)
    report['matched_fltrust_tm_A'] = matched('fltrust', 'trimmed_mean_f1', 'A')
    report['matched_fltrust_tm_B'] = matched('fltrust', 'trimmed_mean_f1', 'B')
    # all rule pairs (architecture A) and top-ranked rule per period
    rA = [m for m, a in rules if a == 'A']; pairs = {}
    for m1, m2 in itertools.combinations(rA, 2):
        n = 0; rev = 0; mat_first = 0
        for c in cells:
            k1, k2 = c + (m1, 'A'), c + (m2, 'A')
            if k1 not in R or k2 not in R: continue
            n += 1; dE = stats[k1]['period_a']['oracle_recall'] - stats[k2]['period_a']['oracle_recall']; dL = stats[k1]['period_b']['oracle_recall'] - stats[k2]['period_b']['oracle_recall']
            rev += abs(dE) > MAT and abs(dL) > MAT and (dE > 0) != (dL > 0)
        pairs[f'{m1}~{m2}'] = dict(cells=n, material_reversals=int(rev))
    report['rule_pairs_A'] = pairs
    top = []
    for c in cells:
        ks = [c + (m, 'A') for m in rA if c + (m, 'A') in R]
        if len(ks) < len(rA): continue
        def best(p):
            v = sorted(((stats[k][p]['oracle_recall'], k[3]) for k in ks), reverse=True)
            return v[0][1] if v[0][0] - v[1][0] > MAT else 'tie'
        top.append(dict(cell='|'.join(map(str, c)), top_earlier=best('period_a'), top_later=best('period_b')))
    report['top_rule_A'] = dict(cells=len(top), differs=sum(t['top_earlier'] != t['top_later'] and 'tie' not in (t['top_earlier'], t['top_later']) for t in top),
                                top_earlier=dict(collections.Counter(t['top_earlier'] for t in top)), top_later=dict(collections.Counter(t['top_later'] for t in top)))
    json.dump(report, open(OUT / f'{ds}-generality.json', 'w'), indent=1)
    slim = {k: v for k, v in report.items() if k not in ('matched_fltrust_tm_A', 'matched_fltrust_tm_B')}
    for k in ('matched_fltrust_tm_A', 'matched_fltrust_tm_B'):
        slim[k] = report[k]['aggregate'] if report[k] else None
    print(json.dumps(slim, indent=1))

if __name__ == '__main__':
    main()
