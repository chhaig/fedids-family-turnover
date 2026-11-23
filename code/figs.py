import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.spines.top':False,'axes.spines.right':False})
U=json.load(open('unified_rescore.json')); S=json.load(open('summary.json'))
main=[];seen=set()
for o in U:
  if o['matrix'] in ('cross-testbed-adaptive-v1-matrix','strong-core-baseline-v1-matrix','nf-unsw-study-matrix-v1') and (o['matrix'],o['run']) not in seen:
    seen.add((o['matrix'],o['run'])); main.append(o)
M={'fltrust':('FLTrust','#2a78d6','o'),'trimmed_mean_f1':('Trimmed mean','#eb6834','s'),'fedavg':('FedAvg','#1baf7a','^'),
   'fltrust_server':('FLTrust+server','#eda100','D'),'root_only':('Root only','#52514e','x'),'boba':('BOBA','#e87ba4','v'),
   'fedval_paper_parameters':('FedVal','#6250d6','P'),'fedval_author_scoring':('FedVal (author scoring)','#6250d6','*')}
T={'nf':'NF-ToN-IoT-v3','cic':'CIC-IDS2017','iot':'IoT-23','unsw':'NF-UNSW-NB15-v3'}
fig,ax=plt.subplots(1,4,figsize=(7.2,2.2))
for i,ds in enumerate(['nf','cic','iot','unsw']):
  a=ax[i]; a.plot([0,100],[0,100],color='#c3c2b7',lw=0.8,zorder=0)
  for m,(lab,c,mk) in M.items():
    L=[o for o in main if o['dataset']==ds and o['method']==m]
    if not L: continue
    a.scatter([100*o['period_a']['oracle_recall'] for o in L],[100*o['period_b']['oracle_recall'] for o in L],s=14,c=c,marker=mk,label=lab,linewidths=0.6 if mk in 'xP*' else 0,alpha=.85,edgecolors='white' if mk not in 'x' else None)
  a.set_xlim(-3,103);a.set_ylim(-3,103);a.set_title(T[ds],fontsize=8);a.set_xlabel('Recall at 1% FPR, earlier period (%)' if i==0 else 'Earlier period (%)')
  if i==0: a.set_ylabel('Later period (%)')
h,l=[],[]
for a in ax:
  for hh,ll in zip(*a.get_legend_handles_labels()):
    if ll not in l: h.append(hh);l.append(ll)
fig.legend(h,l,loc='lower center',ncol=8,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.04),handletextpad=0.2,columnspacing=0.8)
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig('fig1_oracle_recall.png',dpi=300); fig.savefig('fig1_oracle_recall.pdf')
# Fig 2
R=S['nf_matched']['rows']
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
mk={'clean':'o','scaled_flip_a':'s','scaled_flip_b':'^'}; cl={29:'#2a78d6',43:'#eb6834'}
for r in R:
  ax[0].scatter(100*(r['fltA']-r['tmA']),100*(r['fltB']-r['tmB']),marker=mk[r['cond']],c=cl[r['seed']],s=22,edgecolors='white',linewidths=0.5)
  ax[1].scatter(100*r['pw'],100*r['xss'],marker=mk[r['cond']],c=cl[r['seed']],s=22,edgecolors='white',linewidths=0.5)
for a in ax: a.axhline(0,color='#c3c2b7',lw=0.8,zorder=0); a.axvline(0,color='#c3c2b7',lw=0.8,zorder=0)
ax[0].set_xlabel('FLTrust − TM, earlier period (pp)'); ax[0].set_ylabel('FLTrust − TM, later period (pp)')
ax[0].set_title('(a) Pooled recall at 1% FPR',fontsize=8)
ax[1].set_xlabel('Password attacks (pp)'); ax[1].set_ylabel('XSS attacks (pp)'); ax[1].set_title('(b) Earlier period, by attack family',fontsize=8)
from matplotlib.lines import Line2D
hs=[Line2D([],[],marker=mk[k],ls='',color='#52514e',label=v) for k,v in [('clean','No attack'),('scaled_flip_a','Scaled flip, client .152'),('scaled_flip_b','Scaled flip, client .195')]]+[Line2D([],[],marker='o',ls='',color=cl[s],label=f'Optimization seed {s}') for s in (29,43)]
fig.legend(handles=hs,loc='lower center',ncol=5,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.09,1,1)); fig.savefig('fig2_nf_matched.png',dpi=300); fig.savefig('fig2_nf_matched.pdf')
# Fig 3 historical threshold FPR
fig,ax=plt.subplots(figsize=(4.2,2.5))
pos=0; ticks=[]; labs=[]
rng=np.random.default_rng(0)
for ds in ['nf','cic','iot','unsw']:
  for p,pl in [('period_a','earlier'),('period_b','later')]:
    v=np.array([100*o[p]['hist_fpr'] for o in main if o['dataset']==ds]); v=np.clip(v,0.01,None)
    ax.scatter(pos+rng.uniform(-.18,.18,len(v)),v,s=6,c='#2a78d6',alpha=.6,linewidths=0)
    ticks.append(pos); labs.append(('NF-ToN' if ds=='nf' else 'CIC' if ds=='cic' else 'IoT-23' if ds=='iot' else 'UNSW')+'\n'+pl); pos+=1
  pos+=0.5
ax.set_yscale('log'); ax.axhline(1,color='#e34948',lw=0.9,ls='--'); ax.text(-0.4,1.25,'1% target',color='#52514e',fontsize=7,ha='left')
ax.set_xticks(ticks); ax.set_xticklabels(labs,fontsize=5.5); ax.set_ylabel('FPR at historical threshold (%)')
ax.set_yticks([0.01,0.1,1,10,100]); ax.set_yticklabels(['≤0.01','0.1','1','10','100'])
fig.tight_layout(); fig.savefig('fig3_threshold_fpr.png',dpi=300); fig.savefig('fig3_threshold_fpr.pdf')
print('ok')
