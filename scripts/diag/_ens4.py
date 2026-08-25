"""Verify 4-seed cornerattn ensemble from existing test_predictions.npz"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod
HOME=os.path.expanduser('~')
ROOT=os.path.join(HOME,'project-107-rank')
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from src.utils import split_by_expr, ranking_metrics

static_p=[]; dynamic_p=[]
for prefix in ['data/delivery1','data/delivery2']:
    for b in ['batch1','batch2','batch3']:
        sp=f'{prefix}/{b}/circuit_static.parquet'; dp=f'{prefix}/{b}/timing_arcs.parquet'
        if os.path.exists(sp) and os.path.exists(dp): static_p.append(sp); dynamic_p.append(dp); continue
        for t in ['circuit_static','timing_arcs']:
            for p in sorted(gb_mod.glob(f'{prefix}/{b}/{t}_part*.parquet')):
                (static_p if 'static' in p else dynamic_p).append(p)
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

dirs=['rank','seed123','seed2024','seed456']
all_preds=[]; targets=None
for d in dirs:
    data=np.load(os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz'))
    all_preds.append(data['preds']); targets=data['targets']

ens=np.mean(all_preds,axis=0)
mn=min(len(test_dyn),len(ens)); tdyn=test_dyn.iloc[:mn]
rk=ranking_metrics(tdyn,ens[:mn],targets[:mn])
hi=rk.get('hi_spread',{}); pa=rk['pair_acc']
print(f"4-seed ensemble (rank+seed123+2024+456):")
print(f"  hi_regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
print(f"  pair: <2%={pa['<2%'][0]:.0f}% 2-5%={pa['2-5%'][0]:.0f}% 5-10%={pa['5-10%'][0]:.0f}% >10%={pa['>10%'][0]:.0f}%")

# Also print single-seed stats for comparison
for d in dirs:
    data=np.load(os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz'))
    rk=ranking_metrics(test_dyn.iloc[:mn],data['preds'][:mn],data['targets'][:mn])
    hi=rk.get('hi_spread',{})
    print(f"  {d}: regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}%")
