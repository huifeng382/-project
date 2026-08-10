"""2-base ensemble: newwave_base(seed42) + b123(seed123)"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod
HOME=os.path.expanduser('~')
ROOT=os.path.join(HOME,'project-107-newwave_base')
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from src.utils import split_by_expr, ranking_metrics

static_p=[]; dynamic_p=[]
for prefix in ['data/delivery1','data/delivery2']:
    for b in ['batch1','batch2','batch3']:
        sp=f'{prefix}/{b}/circuit_static.parquet'
        dp=f'{prefix}/{b}/timing_arcs.parquet'
        if os.path.exists(sp) and os.path.exists(dp): static_p.append(sp); dynamic_p.append(dp); continue
        sparts=sorted(gb_mod.glob(f'{prefix}/{b}/circuit_static_part*.parquet'))
        dparts=sorted(gb_mod.glob(f'{prefix}/{b}/timing_arcs_part*.parquet'))
        if sparts and dparts: static_p.extend(sparts); dynamic_p.extend(dparts)
dyn_all=pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn_all=dyn_all.dropna(subset=['circuit_id','DELAY']); dyn_all['circuit_id']=dyn_all['circuit_id'].astype(str)
dyn_all=dyn_all[(dyn_all['DELAY']>1e-12)&(dyn_all['DELAY']<1e-8)]
st=pd.concat([pd.read_parquet(p) for p in static_p])
for c in ['candidate','candidate_id']:
    if c in st.columns: st=st.rename(columns={c:'circuit_id'})
st['circuit_id']=st['circuit_id'].astype(str)
four=set()
for _,r in st.iterrows():
    try:
        pins=json.loads(r['input_pins_json']) if isinstance(r['input_pins_json'],str) else r['input_pins_json']
        if sorted(pins)==['a','b','c','d']: four.add(r['circuit_id'])
    except: pass
dyn_all=dyn_all[dyn_all['circuit_id'].isin(four)]
if 'expr' not in dyn_all.columns:
    dyn_all['expr']=dyn_all['circuit_id'].map(dict(zip(st['circuit_id'],st['expr'].astype(str))))
ids=dyn_all['circuit_id'].unique().tolist()
id2e=dict(zip(dyn_all['circuit_id'].astype(str),dyn_all['expr'].astype(str)))
_,_,test_ids=split_by_expr(ids,id2e,seed=42)
test_dyn=dyn_all[dyn_all['circuit_id'].isin(test_ids)].reset_index(drop=True)

# Load both midpoint predictions — for base42, best is ep224 (only 1 midpoint used). For b123, best is mid ep254.
# Use the best available: base42 uses mid (ep didn't matter much), b123 uses mid
# Actually, both reload the best midpoint automatically in SUMMARY. The test_predictions.npz is from the midpoint!
for label, d in [('base42-mid','newwave_base'),('b123-mid','b123')]:
    data=np.load(os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz'))
    print(f"{label}: {len(data['preds'])} preds, median_rel={np.median(np.abs(data['preds']-data['targets'])/data['targets'])*100:.1f}%")

p1=np.load(os.path.join(HOME,'project-107-newwave_base/outputs/test_predictions.npz'))
p2=np.load(os.path.join(HOME,'project-107-b123/outputs/test_predictions.npz'))
ens=np.mean([p1['preds'],p2['preds']],axis=0); tgt=p1['targets']
mn=min(len(test_dyn),len(ens)); tdyn=test_dyn.iloc[:mn]
rk=ranking_metrics(tdyn,ens[:mn],tgt[:mn])
hi=rk.get('hi_spread',{}); pa=rk['pair_acc']
print(f"\n=== 2-base Ensemble (midpoint) ===")
print(f"hi_regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
print(f"pair: <2%={pa['<2%'][0]:.0f}% 2-5%={pa['2-5%'][0]:.0f}% 5-10%={pa['5-10%'][0]:.0f}% >10%={pa['>10%'][0]:.0f}%")
