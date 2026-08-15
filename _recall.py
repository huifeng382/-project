"""recall@K 评估：A严格(真#1进前K) / B宽松(前K有真前K之一)，K=2/3。

读 8 个 cornerattn seed 的 test_predictions.npz，输出各单 seed + 关键集成的 recall@K
（全局 与 spread>10% 两组口径），并标注每档的有效组数 n（仅 size>=K+1 的非平凡组）。
"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod
HOME=os.path.expanduser('~')
ROOT=os.path.join(HOME,'project-107-rank')
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from src.utils import split_by_expr, ranking_metrics

# --- 加载 test 动态数据（与 _trim8.py 一致） ---
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

# --- 读 8 个 seed 的 npz ---
dirs=['rank','seed123','seed2024','seed456','seed1357','seed2468','seed3579','seed9012']
preds_by={}; targets=None
for d in dirs:
    p=os.path.join(HOME,f'project-107-{d}/outputs/test_predictions.npz')
    data=np.load(p); preds_by[d]=data['preds']; targets=data['targets']
mn=min(len(test_dyn), len(targets))

def _fmt(rec):
    s=[]
    for K in sorted(rec):
        st=rec[K]['strict']; le=rec[K]['lenient']
        s.append(f"@K={K} A={st['hit_pct']*100:.1f}%(n={st['n']}) B={le['hit_pct']*100:.1f}%(n={le['n']})")
    return '  '.join(s)

def show(name, preds):
    rk=ranking_metrics(test_dyn.iloc[:mn], preds[:mn], targets[:mn])
    hi=rk.get('hi_spread',{})
    print(f"\n{name}")
    print(f"  全局     遗憾={rk['regret_pct']:.2f}% top1={rk['top1_acc']*100:.1f}%  recall: {_fmt(rk.get('recall_at_k',{}))}")
    print(f"  spread>10% 遗憾={hi.get('regret_pct',0):.2f}% top1={hi.get('top1_acc',0)*100:.1f}%  recall: {_fmt(hi.get('recall_at_k',{}))}")

for d in dirs:
    show(d, preds_by[d])

# 关键集成
show('TOP-2 (2468+456)', np.mean([preds_by['seed2468'],preds_by['seed456']],axis=0))
show('TOP-3 (2468+456+1357)', np.mean([preds_by['seed2468'],preds_by['seed456'],preds_by['seed1357']],axis=0))
show('TOP-4 (2468+456+1357+2024)', np.mean([preds_by['seed2468'],preds_by['seed456'],preds_by['seed1357'],preds_by['seed2024']],axis=0))
show('全8平均', np.mean([preds_by[d] for d in dirs],axis=0))
