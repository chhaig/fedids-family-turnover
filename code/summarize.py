import json, collections, statistics as st
U = json.load(open('unified_rescore.json')); SW = json.load(open('pool_swap.json'))
MAT = 0.01
def dedup(ms):
    seen = set(); out = []
    for o in U:
        if o['matrix'] in ms and (o['matrix'], o['run']) not in seen:
            seen.add((o['matrix'], o['run'])); out.append(o)
    return out
R = {}
# 1. NF matched panel
nf = dedup({'nf-root-seed-v1-matrix'}); P = collections.defaultdict(dict)
for o in nf: P[(o['condition'], o['seed'], o['run'].split('-')[-1])][o['method']] = o
sw = {(o['matrix'], o['run']): o['swap'] for o in SW}
c = collections.Counter(); rows = []
for k, v in sorted(P.items()):
    a, b = v['fltrust'], v['trimmed_mean_f1']
    dA = a['period_a']['oracle_recall'] - b['period_a']['oracle_recall']; dB = a['period_b']['oracle_recall'] - b['period_b']['oracle_recall']
    aA = a['period_a']['auc'] - b['period_a']['auc']; aB = a['period_b']['auc'] - b['period_b']['auc']
    mrev = abs(dA) > MAT and abs(dB) > MAT and (dA > 0) != (dB > 0)
    arev = (aA > 0) != (aB > 0)
    sa = sw[('nf-root-seed-v1-matrix', a['run'])]; sb = sw[('nf-root-seed-v1-matrix', b['run'])]
    s = {x: sa[x] - sb[x] for x in sa}
    def sg(x): return 0 if abs(s[x]) <= MAT else (1 if s[x] > 0 else -1)
    att = sg('ben_a|att_a') == sg('ben_b|att_a') != 0 and sg('ben_a|att_b') == sg('ben_b|att_b') != 0 and sg('ben_a|att_a') != sg('ben_a|att_b')
    ben = sg('ben_a|att_a') == sg('ben_a|att_b') != 0 and sg('ben_b|att_a') == sg('ben_b|att_b') != 0 and sg('ben_a|att_a') != sg('ben_b|att_a')
    pw = a['period_a']['family']['password'][2] - b['period_a']['family']['password'][2]
    xs = a['period_a']['family']['xss'][2] - b['period_a']['family']['xss'][2]
    c.update({'pairs': 1, 'material_rev': mrev, 'auc_rev': arev, 'attack_driven': att, 'benign_driven': ben, 'flt_wins_A': dA > MAT, 'tm_wins_B': dB < -MAT,
              'password_flt_better': pw > MAT, 'xss_within1pp': abs(xs) <= MAT, 'flt_B_below1pct': a['period_b']['oracle_recall'] < .01})
    rows.append(dict(cond=k[0], seed=k[1], root=k[2], fltA=a['period_a']['oracle_recall'], tmA=b['period_a']['oracle_recall'], fltB=a['period_b']['oracle_recall'], tmB=b['period_b']['oracle_recall'],
                     aucA=(a['period_a']['auc'], b['period_a']['auc']), aucB=(a['period_b']['auc'], b['period_b']['auc']), pw=pw, xss=xs, swap=s))
R['nf_matched'] = dict(counts=dict(c), rows=rows)
# 2. cross-testbed panel summary per dataset x method
main = dedup({'cross-testbed-adaptive-v1-matrix', 'strong-core-baseline-v1-matrix', 'nf-unsw-study-matrix-v1'})
g = collections.defaultdict(list)
for o in main: g[(o['dataset'], o['method'])].append(o)
summ = {}
for k, L in g.items():
    q = lambda p, key: [o[p][key] for o in L]
    summ['|'.join(k)] = {p: {key: [min(q(p, key)), st.median(q(p, key)), max(q(p, key))] for key in ('auc', 'oracle_recall', 'hist_recall', 'hist_fpr')} for p in ('period_a', 'period_b')} | {'n': len(L)}
R['panel'] = summ
# families per dataset
fam = {}
for o in main:
    if o['dataset'] not in fam:
        fam[o['dataset']] = {p: {f: v[0] for f, v in o[p]['family'].items()} for p in ('period_a', 'period_b')} | {'days': {p: o[p]['days'] for p in ('period_a', 'period_b')}}
R['families'] = fam
# material pairwise reversals in panel (fltrust vs trimmed mean, same condition)
pp = collections.defaultdict(dict)
for o in main: pp[(o['dataset'], o['condition'], o['seed'])][o['method']] = o
cnt = collections.Counter()
for k, v in pp.items():
    if 'fltrust' in v and 'trimmed_mean_f1' in v:
        a, b = v['fltrust'], v['trimmed_mean_f1']
        dA = a['period_a']['oracle_recall'] - b['period_a']['oracle_recall']; dB = a['period_b']['oracle_recall'] - b['period_b']['oracle_recall']
        cnt[(k[0], 'pairs')] += 1
        cnt[(k[0], 'material_rev')] += abs(dA) > MAT and abs(dB) > MAT and (dA > 0) != (dB > 0)
        cnt[(k[0], 'B_both_below1pct')] += a['period_b']['oracle_recall'] < .01 and b['period_b']['oracle_recall'] < .01
        cnt[(k[0], 'B_gap_within1pp')] += abs(dB) <= MAT
R['panel_pairs'] = {f'{a}|{b}': n for (a, b), n in cnt.items()}
json.dump(R, open('summary.json', 'w'), indent=1)
print(json.dumps(R['nf_matched']['counts'])); print(json.dumps(R['panel_pairs']))
for k in sorted(summ): 
    s = summ[k]; print(k, s['n'], {p: {kk: [round(x, 3) for x in vv] for kk, vv in s[p].items()} for p in ('period_a', 'period_b')})
