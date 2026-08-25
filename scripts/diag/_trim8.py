"""Trimmed mean ensemble over all 8 cornerattn seeds (4 old 13.6 + 4 new 14.0).

Reads every seed's test_predictions.npz, prints single-seed hi_spread regret,
reproduces the old-4 trim (current best 2.08%) as a sanity baseline, then does a
full trim sweep over all 8 (keep top-K by regret, K=8..2).
"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod
HOME=os.path.expanduser('~')
ROOT=os.path.join(HOME,'project-107-rank')
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from src.utils import split_by_expr, ranking_metrics

# --- load test dynamic data (identical to _trim.py) ---
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

# --- all 8 cornerattn seeds ---
dirs=['rank','seed123','seed2024','seed456','seed1357','seed2468','seed3579','seed9012']
old4=[0,1,2,3]  # indices of the 13.6 seeds

preds_by_dir={}; regret_by_dir={}; targets=None; missing=[]
for i,d in enumerate(dirs):
    p=os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz')
    if not os.path.exists(p):
        missing.append(d); preds_by_dir[d]=None; regret_by_dir[d]=1e9
        print(f"  {d}: [MISSING npz]"); continue
    data=np.load(p)
    preds_by_dir[d]=data['preds']; targets=data['targets']
    mn=min(len(test_dyn),len(data['preds']))
    rk=ranking_metrics(test_dyn.iloc[:mn],data['preds'][:mn],data['targets'][:mn])
    hi=rk.get('hi_spread',{})
    regret_by_dir[d]=hi.get('regret_pct',100)
    print(f"  {d}: regret={regret_by_dir[d]:.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
if missing:
    print(f"\n[WARN] missing: {missing}")

def try_ens(names, label):
    keep=[d for d in names if preds_by_dir[d] is not None]
    if not keep: print(f"{label}: no available seeds"); return
    ens=np.mean([preds_by_dir[d] for d in keep],axis=0)
    mn=min(len(test_dyn),len(ens)); tdyn=test_dyn.iloc[:mn]
    rk=ranking_metrics(tdyn,ens[:mn],targets[:mn])
    hi=rk.get('hi_spread',{}); pa=rk['pair_acc']
    print(f"\n{label}: regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
    print(f"  seeds: {keep}  regrets {[f'{regret_by_dir[d]:.2f}%' for d in keep]}")
    print(f"  pair: <2%={pa['<2%'][0]:.0f}% 2-5%={pa['2-5%'][0]:.0f}% 5-10%={pa['5-10%'][0]:.0f}% >10%={pa['>10%'][0]:.0f}%")

# --- sanity: reproduce old-4 trim (current best 2.08%) ---
print("\n=== old-4 baseline (should reproduce Trim1=2.08%) ===")
old_names=[dirs[i] for i in old4]
try_ens(old_names, "old-4 all")
order_old=sorted(old_names, key=lambda d: regret_by_dir[d])
try_ens(order_old[:3], f"old-4 Trim1 (drop {order_old[3]})")
try_ens(order_old[:2], f"old-4 Trim2 (keep {order_old[0]}+{order_old[1]})")

# --- full trim sweep over all 8 ---
print("\n=== all-8 trim sweep (keep top-K by regret) ===")
order_all=sorted(dirs, key=lambda d: regret_by_dir[d])
for K in range(len(dirs),1,-1):
    try_ens(order_all[:K], f"all-8 keep top-{K} (trim {len(dirs)-K} worst)")
