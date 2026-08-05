"""Mid-training report: load best_model.pt from each run, evaluate on test set.
Run from a project directory that has data/ subdirectory (e.g., project-107-rank)."""
import json, torch, pandas as pd, numpy as np, sys, os, glob as gb_mod
# Find project root: the directory containing data/delivery1
_here = os.path.abspath(os.path.dirname(__file__))
for _root in [_here, os.path.dirname(_here), os.path.expanduser('~/exp107')]:
    if os.path.isdir(f'{_root}/data/delivery1'):
        os.chdir(_root); sys.path.insert(0, _root); break
import config
config.USE_TRANSISTOR_WAVE=True; config.USE_STRUCT_PRIOR=True; config.USE_CORNER_ATTN=True
config.USE_PARASITIC_CAPS=False; config.USE_SUPPLY_NOISE=False; config.WAVE_AGG_RICH=False
from src.model import DelayGNN
from src.graph_builder import rebuild_gate_types, GATE_TYPES
from src.utils import split_by_expr, ranking_metrics

static_p = []; dynamic_p = []
for prefix in ['data/delivery1','data/delivery2']:
    for b in ['batch1','batch2','batch3']:
        sp=f'{prefix}/{b}/circuit_static.parquet'
        dp=f'{prefix}/{b}/timing_arcs.parquet'
        if os.path.exists(sp) and os.path.exists(dp):
            static_p.append(sp); dynamic_p.append(dp); continue
        sparts=sorted(gb_mod.glob(f'{prefix}/{b}/circuit_static_part*.parquet'))
        dparts=sorted(gb_mod.glob(f'{prefix}/{b}/timing_arcs_part*.parquet'))
        if sparts and dparts: static_p.extend(sparts); dynamic_p.extend(dparts)

dyn_all = pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn_all = dyn_all.dropna(subset=['circuit_id','DELAY'])
dyn_all['circuit_id']=dyn_all['circuit_id'].astype(str)
dyn_all = dyn_all[(dyn_all['DELAY']>1e-12)&(dyn_all['DELAY']<1e-8)]
st = pd.concat([pd.read_parquet(p) for p in static_p])
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
if 'expr' not in dyn_all.columns and 'expr' in st.columns:
    dyn_all['expr']=dyn_all['circuit_id'].map(dict(zip(st['circuit_id'],st['expr'].astype(str))))
ids=dyn_all['circuit_id'].unique().tolist()
id2e=dict(zip(dyn_all['circuit_id'].astype(str),dyn_all['expr'].astype(str))) if 'expr' in dyn_all.columns else None
_,_,test_ids=split_by_expr(ids,id2e,seed=42)
test_dyn=dyn_all[dyn_all['circuit_id'].isin(test_ids)].reset_index(drop=True)

# Rebuild gate types — must account for ALL deliveries (training sees all circuits)
allct=set()
static_all=[]
for prefix in ['data/delivery1','data/delivery2']:
    for b in ['batch1','batch2','batch3']:
        sp=f'{prefix}/{b}/circuit_static.parquet'
        if os.path.exists(sp): static_all.append(sp); continue
        for p in gb_mod.glob(f'{prefix}/{b}/circuit_static_part*.parquet'):
            static_all.append(p)
for p in static_all:
    df=pd.read_parquet(p)
    for c in df['cell_types_json']:
        a=json.loads(c) if isinstance(c,str) else c
        if a: allct.update(a)
print(f"Loading {len(static_all)} static files: {static_all}")
allct=set()
for p in static_all:
    df=pd.read_parquet(p)
    ct=set()
    for c in df['cell_types_json']:
        a=json.loads(c) if isinstance(c,str) else c
        if a: ct.update(a)
    print(f"  {p}: {len(df)} circuits, {len(ct)} gate types")
    allct.update(ct)
rebuild_gate_types(list(allct))
num_gt=len(GATE_TYPES)
print(f"gate_types total: {num_gt}, test rows: {len(test_dyn)}")

# Eval each variant (models in ~/project-107-X)
HOME = os.path.expanduser('~')
for d in ['hard5','hard5w2','hard10','hard10w2']:
    mp=f'{HOME}/project-107-{d}/outputs/best_model.pt'
    if not os.path.exists(mp):
        print(f"\n{d}: best_model.pt not found")
        continue
    ckpt = torch.load(mp, map_location='cpu', weights_only=False)
    num_gt_from_ckpt = ckpt['gate_embed.weight'].shape[0]
    model=DelayGNN(in_dim=17,hidden_dim=256,num_layers=6,dropout=0.3,num_gate_types=num_gt_from_ckpt,gate_embed_dim=32)
    model.load_state_dict(ckpt)
    model.eval()
    # Eval on test set — use a mini DataLoader
    from src.data_loader import DelayDataset
    from torch_geometric.loader import DataLoader
    print(f"  Building test dataset ({len(test_ids)} circuits, reusing cache)...", flush=True)
    ds_test=DelayDataset(static_p,dynamic_p,test_ids,scaler=None,cache_dir=f'{HOME}/project-107-{d}/cache107{d}')
    print(f"  Dataset ready, loading predictions...", flush=True)
    loader=DataLoader(ds_test,batch_size=80,shuffle=False)
    preds=[]; targets=[]
    device='cpu'
    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            out,_=model(data.x,data.edge_index,data.batch,data.corner_cond,data.circuit_sig,getattr(data,'struct_prior',None))
            preds.append(out.cpu().numpy()); targets.append(data.y.cpu().numpy())
    preds=np.concatenate(preds); targets=np.concatenate(targets)
    if len(preds)!=len(test_dyn):
        print(f"\n{d}: len mismatch preds={len(preds)} test_dyn={len(test_dyn)}")
        # try to align by taking first N
        mn=min(len(preds),len(test_dyn)); preds=preds[:mn]; targets=targets[:mn]; tdyn=test_dyn.iloc[:mn]
    else:
        tdyn=test_dyn
    # Metrics
    rk=ranking_metrics(tdyn,preds,targets)
    hi=rk.get('hi_spread',{})
    print(f"\n{d}:")
    print(f"  Median Rel={np.median(np.abs(preds-targets)/targets)*100:.1f}%  Mean Abs={np.mean(np.abs(preds-targets))*1e12:.1f}ps")
    if hi.get('n',0)>0:
        print(f"  Hi-spread Spearman={hi['spearman']:.3f}  regret={hi['regret_pct']:.2f}%  top1={hi['top1_acc']*100:.1f}%  capture={hi['captured_pct']:.1f}%")
    pa=rk['pair_acc']
    print(f"  Pairwise: <2%:{pa['<2%'][0]:.0f}%  2-5%:{pa['2-5%'][0]:.0f}%  5-10%:{pa['5-10%'][0]:.0f}%  >10%:{pa['>10%'][0]:.0f}%")
