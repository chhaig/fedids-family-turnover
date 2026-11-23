import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
R=json.load(open('family_decomp.json'))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6),sharey=True)
pi=np.linspace(0,1,101); cl={29:'#2a78d6',43:'#eb6834'}; ls={'clean':'-','scaled_flip_a':'--','scaled_flip_b':':'}
for a,bp,t in zip(ax,('period_a','period_b'),('(a) Benign pool and threshold: earlier period','(b) Benign pool and threshold: later period')):
  for r in R:
    m=r['mix'][bp]; D=100*((1-pi)*m['d_earlier_mix']+pi*m['d_later_mix'])
    a.plot(100*pi,D,color=cl[r['seed']],ls=ls[r['cond']],lw=0.9,alpha=.8)
  a.axhline(0,color='#c3c2b7',lw=0.8,zorder=0); a.set_title(t,fontsize=8); a.set_xlabel('Backdoor share of attack mass (%)')
ax[0].set_ylabel('FLTrust − TM recall at 1% FPR (pp)')
from matplotlib.lines import Line2D
h=[Line2D([],[],color='#52514e',ls=ls[k],label=v) for k,v in [('clean','No attack'),('scaled_flip_a','Scaled flip, client .152'),('scaled_flip_b','Scaled flip, client .195')]]+[Line2D([],[],color=cl[s],label=f'Optimization seed {s}') for s in (29,43)]
fig.legend(handles=h,loc='lower center',ncol=5,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.09,1,1)); fig.savefig('fig2_family_mix.png',dpi=300); fig.savefig('fig2_family_mix.pdf')
