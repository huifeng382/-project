"""Test cross-ensemble stability: base_ep200 + hard10_best"""
import numpy as np, os, pandas as pd, json, torch, sys, glob as gb_mod
HOME=os.path.expanduser('~')
ROOT=os.path.join(HOME,'project-107-newwave_base')
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from src.utils import split_by_expr, ranking_metrics
from src.data_loader import DelayDataset
from src.model import DelayGNN
from torch_geometric.loader import DataLoader

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

def eval_ckpt(path, tag):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    m=DelayGNN(17,256,6,0.3,ckpt['gate_embed.weight'].shape[0],32)
    m.load_state_dict(ckpt); m.eval()
    ds=DelayDataset(static_p,dynamic_p,test_ids,scaler=None,cache_dir=f'cx2_{tag}')
    loader=DataLoader(ds,batch_size=80,shuffle=False)
    preds=[]; targets=[]
    with torch.no_grad():
        for data in loader:
            out,_=m(data.x,data.edge_index,data.batch,data.corner_cond,data.circuit_sig,getattr(data,'struct_prior',None))
            preds.append(out.cpu().numpy()); targets.append(data.y.cpu().numpy())
    return np.concatenate(preds),np.concatenate(targets)

pairs=[
    ('midpoint_ep100','best_model', 'ep100+hard_best'),
    ('midpoint_ep200','best_model', 'ep200+hard_best'),
    ('best_model','best_model', 'base_best+hard_best'),
]
for b_ckpt, h_ckpt, label in pairs:
    bp,bt=eval_ckpt(os.path.join(HOME,f'project-107-newwave_base/outputs/{b_ckpt}.pt'),f'b{label}')
    hp,ht=eval_ckpt(os.path.join(HOME,f'project-107-newwave_hard10/outputs/{h_ckpt}.pt'),f'h{label}')
    ens=np.mean([bp,hp],axis=0)
    mn=min(len(test_dyn),len(ens)); tdyn=test_dyn.iloc[:mn]
    rk=ranking_metrics(tdyn,ens[:mn],bt[:mn])
    hi=rk.get('hi_spread',{})
    print(f"{label:25s}: regret={hi.get('regret_pct',0):.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
