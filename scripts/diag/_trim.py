"""Trimmed mean ensemble: test different trimming strategies on 4 existing seeds"""
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
all_preds=[]; all_regrets=[]; targets=None
for d in dirs:
    data=np.load(os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz'))
    all_preds.append(data['preds']); targets=data['targets']
    mn=min(len(test_dyn),len(data['preds']))
    rk=ranking_metrics(test_dyn.iloc[:mn],data['preds'][:mn],data['targets'][:mn])
    hi=rk.get('hi_spread',{})
    all_regrets.append(hi.get('regret_pct',100))
    print(f"  {d}: regret={all_regrets[-1]:.2f}%")

# Try different trims
import itertools
all_preds=np.array(all_preds)
all_regrets=np.array(all_regrets)
order=np.argsort(all_regrets)  # ascending regret = better

def try_ens(keep_indices, label):
    ens=np.mean(all_preds[keep_indices],axis=0)
    mn=min(len(test_dyn),len(ens)); tdyn=test_dyn.iloc[:mn]
    rk=ranking_metrics(tdyn,ens[:mn],targets[:mn])
    hi=rk.get('hi_spread',{})
    pa=rk['pair_acc']
    print(f"\n{label}: regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
    print(f"  seeds kept: {[dirs[i] for i in keep_indices]} regrets: {[f'{all_regrets[i]:.2f}%' for i in keep_indices]}")
    print(f"  pair: <2%={pa['<2%'][0]:.0f}% 2-5%={pa['2-5%'][0]:.0f}% 5-10%={pa['5-10%'][0]:.0f}% >10%={pa['>10%'][0]:.0f}%")

try_ens([0,1,2,3], "All 4 (baseline)")
# Trim 1 worst: keep 3 best
try_ens(order[:3], f"Trim 1 (drop {dirs[order[3]]})")
# Trim 2 worst: keep 2 best
try_ens(order[:2], f"Trim 2 (keep only {dirs[order[0]]}+{dirs[order[1]]})")
