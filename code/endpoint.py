import json,collections
from scipy.stats import beta
U=json.load(open('unified_rescore.json'))
main=[];seen=set()
for o in U:
  if o['matrix'] in ('cross-testbed-adaptive-v1-matrix','strong-core-baseline-v1-matrix','nf-unsw-study-matrix-v1') and (o['matrix'],o['run']) not in seen:
    seen.add((o['matrix'],o['run'])); main.append(o)
ds_seen=set(); out={'support':{},'runs':{}}
for o in main:
  if o['dataset'] in ds_seen: continue
  ds_seen.add(o['dataset'])
  out['support'][o['dataset']]={p:{k:v[1] for k,v in o[p]['owner'].items()} for p in ('period_a','period_b')}
c=collections.Counter()
for o in main:
  for p in ('period_a','period_b'):
    ex=cex=cin=tot=0
    for k,(w,n,f) in o[p]['owner'].items():
      if n==0 or f is None: continue
      tot+=1; kk=round(f*n)
      lo=beta.ppf(.025,kk,n-kk+1) if kk>0 else 0.0
      hi=beta.ppf(.975,kk+1,n-kk) if kk<n else 1.0
      ex+=int(f>0.01); cex+=int(lo>0.01); cin+=int(hi<=0.01)
    key=f"{o['dataset']}|{p}"
    c[key+'|runs']+=1; c[key+"|any_point_exceed"]+=int(ex>0); c[key+"|any_cert_exceed"]+=int(cex>0); c[key+"|all_cert_within"]+=int(cin==tot); c[key+'|endpoints']=tot
out['runs']=dict(c)
json.dump(out,open('endpoint.json','w'),indent=1)
for k,v in out['support'].items(): print(k,v)
for k in sorted(c): print(k,c[k])
# min n for certifying <=1% with 0 FP
n=1
while beta.ppf(.975,1,n)>0.01: n+=1
print('min n zero-FP certify',n)
