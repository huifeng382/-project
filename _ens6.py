"""6-base ensemble + best-3 ensemble: midpoint predictions"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod
HOME=os.path.expanduser('~')
ROOT=os.path.join(HOME,'project-107-newwave_base')
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

dirs={'42':'newwave_base','123':'b123','2024':'b2024','3456':'b3456','5678':'b5678','7890':'b7890'}
all_preds=[]; targets=None
for seed,d in dirs.items():
    data=np.load(os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz'))
    all_preds.append(data['preds']); targets=data['targets']

def ensemble_report(seeds, label):
    idxs=[int(s) if s!='' else i for i,s in enumerate(seeds)]
    preds_list=[all_preds[list(dirs.keys()).index(str(s))] for s in seeds]
    ens=np.mean(preds_list,axis=0)
    mn=min(len(test_dyn),len(ens)); tdyn=test_dyn.iloc[:mn]
    rk=ranking_metrics(tdyn,ens[:mn],targets[:mn])
    hi=rk.get('hi_spread',{}); pa=rk['pair_acc']
    print(f"{label}: regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%  |  "
          f"pair <2%={pa['<2%'][0]:.0f}% 2-5%={pa['2-5%'][0]:.0f}% 5-10%={pa['5-10%'][0]:.0f}% >10%={pa['>10%'][0]:.0f}%")

# All 6
ensemble_report(['42','123','2024','3456','5678','7890'], "6-base")
# Best 3 (lowest regret: 123, 3456, 7890)
ensemble_report(['123','3456','7890'], "best-3(123+3456+7890)")
# Best 4 (123, 3456, 7890, 42)
ensemble_report(['42','123','3456','7890'], "best-4(42+123+3456+7890)")
# Best 5 (drop 5678 only)
ensemble_report(['42','123','2024','3456','7890'], "best-5(drop 5678)")
