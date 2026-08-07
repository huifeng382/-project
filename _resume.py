"""Resume training from best_model.pt. Usage: python _resume.py <proj_dir>"""
import sys, os, torch, pandas as pd, json, numpy as np, hashlib, time, subprocess, glob as gb_mod
PROJ = os.path.abspath(sys.argv[1])
os.chdir(PROJ); sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'src'))
from config import *
from src.utils import set_seed, split_by_expr, ranking_metrics
from src.data_loader import DelayDataset
from src.model import DelayGNN
from src.graph_builder import rebuild_gate_types, GATE_TYPES
from torch_geometric.loader import DataLoader
from torch.optim import Adam

set_seed(TRAIN_SEED)
data_dir = PROJ

# Load data (same as train_sweep)
import glob
static_p = []; dynamic_p = []
for prefix in ['data/delivery1','data/delivery2']:
    for b in ['batch1','batch2','batch3']:
        sp = os.path.join(data_dir, f'{prefix}/{b}/circuit_static.parquet')
        dp = os.path.join(data_dir, f'{prefix}/{b}/timing_arcs.parquet')
        if os.path.exists(sp) and os.path.exists(dp):
            static_p.append(sp); dynamic_p.append(dp); continue
        sparts = sorted(glob.glob(os.path.join(data_dir, f'{prefix}/{b}/circuit_static_part*.parquet')))
        dparts = sorted(glob.glob(os.path.join(data_dir, f'{prefix}/{b}/timing_arcs_part*.parquet')))
        if sparts and dparts: static_p.extend(sparts); dynamic_p.extend(dparts)

dyn = pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn = dyn.dropna(subset=['circuit_id','DELAY']); dyn['circuit_id']=dyn['circuit_id'].astype(str)
dyn = dyn[(dyn['DELAY']>1e-12)&(dyn['DELAY']<1e-8)]
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
dyn=dyn[dyn['circuit_id'].isin(four)]
if 'expr' not in dyn.columns and 'expr' in st.columns:
    dyn['expr']=dyn['circuit_id'].map(dict(zip(st['circuit_id'],st['expr'].astype(str))))
ids=dyn['circuit_id'].unique().tolist()
id2e=dict(zip(dyn['circuit_id'].astype(str),dyn['expr'].astype(str))) if 'expr' in dyn.columns else None
tr,va,te=split_by_expr(ids,id2e,seed=SPLIT_SEED)
print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

# Build datasets (reuse cache)
train_ds=DelayDataset(static_p,dynamic_p,tr,scaler=None,cache_dir=os.path.join(PROJ,'cache107hard5w2'))
val_ds=DelayDataset(static_p,dynamic_p,va,scaler=None,cache_dir=os.path.join(PROJ,'cache107hard5w2'))
samp=train_ds[0]; in_dim=samp.x.shape[1]
print(f"in_dim={in_dim}")

# Load model — use gate_embed shape from checkpoint
ckpt=torch.load(os.path.join(PROJ,'outputs','best_model.pt'),map_location='cpu',weights_only=False)
num_gt_from_ckpt=ckpt['gate_embed.weight'].shape[0]
print(f"Gate types from ckpt: {num_gt_from_ckpt}")
model=DelayGNN(in_dim=in_dim,hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,dropout=DROPOUT,
               num_gate_types=num_gt_from_ckpt,gate_embed_dim=GATE_EMBED_DIM)
model.load_state_dict(ckpt)
print("Model loaded from best_model.pt")

# Continue training
if RANK_LOSS_W>0:
    from src.utils import GroupedBatchSampler
    sampler=GroupedBatchSampler(train_ds.group_ids,BATCH_SIZE,shuffle=True)
    train_loader=DataLoader(train_ds,batch_sampler=sampler,num_workers=2)
else:
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,num_workers=2)
optimizer=Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=LR_FACTOR,
                                                      patience=LR_PATIENCE,min_lr=LR_MIN,cooldown=LR_COOLDOWN)

# Resume from epoch 304 — set LR to last known
for pg in optimizer.param_groups: pg['lr']=3.13e-06

from src.train_sweep import train_one_epoch, evaluate, _pairwise_rank_loss
best_val_rel=float('inf'); best_val_loss=float('inf'); best_sel=float('inf')
patience_counter=0; plateau_counter=0; val_loss_history=[]; plateau_triggered=False; lr_decayed=True
last_lr=3.13e-06
START_EP=304
print(f"Resuming from epoch {START_EP}...")
t0=time.time()
for ep in range(START_EP, EPOCHS):
    train_loss=train_one_epoch(model,train_loader,optimizer,'cpu',delta=HUBER_DELTA)
    val_loss,val_rel_err,_,_=evaluate(model,val_loader,'cpu')
    current_lr=optimizer.param_groups[0]['lr']
    print(f"Epoch {ep+1:03d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Rel Err: {val_rel_err:.2f}%")
    if scheduler: scheduler.step(val_loss)
    val_loss_history.append(val_loss)
    if val_rel_err<best_val_rel: best_val_rel=val_rel_err
    # checkpoint: smoothed_rel_err
    from config import BEST_MODEL_METRIC, BEST_SMOOTH_WINDOW
    if BEST_MODEL_METRIC=='val_loss': sel=val_loss
    elif BEST_MODEL_METRIC=='smoothed_rel_err':
        sel=float(np.mean(val_loss_history[-BEST_SMOOTH_WINDOW:])) if len(val_loss_history)>=BEST_SMOOTH_WINDOW else val_rel_err
    else: sel=val_rel_err
    if sel<best_sel:
        best_sel=sel
        torch.save(model.state_dict(),os.path.join(PROJ,'outputs','best_model.pt'))
        print(f"  >>> New best model ({BEST_MODEL_METRIC}={sel:.4f}, ValRelErr={val_rel_err:.2f}%)")
    if val_loss<best_val_loss-1e-5: best_val_loss=val_loss; patience_counter=0; plateau_counter=0
    else:
        patience_counter+=1; plateau_counter+=1
        if patience_counter>=PATIENCE:
            print("Early stopping"); break
        if plateau_counter>=PLATEAU_WINDOW and ep+1>=PLATEAU_MIN_EPOCHS and lr_decayed and not plateau_triggered:
            plateau_triggered=True
            print(f">>> Plateau detected, stopping"); break
    # midpoint save
    if SAVE_MIDPOINTS and (ep+1)%MIDPOINT_INTERVAL==0:
        torch.save(model.state_dict(),os.path.join(PROJ,'outputs',f'midpoint_ep{ep+1}.pt'))

elapsed=(time.time()-t0)/60
print(f"Training done in {elapsed:.1f} min, best_val_rel={best_val_rel:.2f}%")
# Final test eval
model.load_state_dict(torch.load(os.path.join(PROJ,'outputs','best_model.pt'),map_location='cpu',weights_only=False))
test_ds=DelayDataset(static_p,dynamic_p,te,scaler=None,cache_dir=os.path.join(PROJ,'cache107hard5w2'))
test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,num_workers=2)
test_loss,test_rel_err,preds,targets=evaluate(model,test_loader,'cpu')
print(f"\nTest Loss: {test_loss:.4f} | Test Rel Err: {test_rel_err:.2f}%")
np.savez(os.path.join(PROJ,'outputs','test_predictions.npz'),preds=preds,targets=targets)
test_dyn=test_ds.dynamic_df.reset_index(drop=True)
# Print ranking
rk=ranking_metrics(test_dyn,preds,targets)
hi=rk.get('hi_spread',{})
print(f"[排序 spread>10%] 组={hi.get('n',0)} Spearman={hi.get('spearman','?'):.3f} 遗憾={hi.get('regret_pct','?'):.2f}% top1={hi.get('top1_acc',0)*100:.1f}%")
pa=rk['pair_acc']
print("[成对] "+" ".join(f"{l}:{pa[l][0]:.0f}%" for l in ['<2%','2-5%','5-10%','>10%']))
