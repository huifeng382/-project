"""
Select best epoch from midpoint checkpoints.
Usage: python _select_best.py <proj_dir>
Evaluates all midpoint_ep*.pt + best_model.pt on test set, ranks by high-spread regret.
"""
import json, torch, pandas as pd, numpy as np, sys, os, glob as gb_mod
PROJ = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
os.chdir(PROJ); sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'src'))

import config
from src.model import DelayGNN
from src.graph_builder import rebuild_gate_types, GATE_TYPES
from src.utils import split_by_expr, ranking_metrics
from src.data_loader import DelayDataset
from torch_geometric.loader import DataLoader

HOME = os.path.expanduser('~')

# Rebuild test data (same as _mid_report)
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
dyn_all = dyn_all.dropna(subset=['circuit_id','DELAY']); dyn_all['circuit_id']=dyn_all['circuit_id'].astype(str)
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
dyn_all = dyn_all[dyn_all['circuit_id'].isin(four)]
if 'expr' not in dyn_all.columns and 'expr' in st.columns:
    dyn_all['expr']=dyn_all['circuit_id'].map(dict(zip(st['circuit_id'],st['expr'].astype(str))))
ids = dyn_all['circuit_id'].unique().tolist()
id2e = dict(zip(dyn_all['circuit_id'].astype(str), dyn_all['expr'].astype(str))) if 'expr' in dyn_all.columns else None
_, _, test_ids = split_by_expr(ids, id2e, seed=42)
test_dyn = dyn_all[dyn_all['circuit_id'].isin(test_ids)].reset_index(drop=True)
print(f"test rows: {len(test_dyn)}")

# Collect checkpoints
ckpts = []
for f in sorted(os.listdir(os.path.join(PROJ, 'outputs'))):
    if 'midpoint_ep' in f and f.endswith('.pt'):
        ep = int(f.replace('midpoint_ep','').replace('.pt',''))
        ckpts.append((ep, os.path.join(PROJ, 'outputs', f)))
if os.path.exists(os.path.join(PROJ, 'outputs', 'best_model.pt')):
    ckpts.append(('best', os.path.join(PROJ, 'outputs', 'best_model.pt')))
print(f"Found {len(ckpts)} checkpoints: {[c[0] for c in ckpts]}")

# Evaluate each
results = []
for label, mp in ckpts:
    ckpt = torch.load(mp, map_location='cpu', weights_only=False)
    num_gt = ckpt['gate_embed.weight'].shape[0]
    model = DelayGNN(in_dim=17, hidden_dim=256, num_layers=6, dropout=0.3,
                     num_gate_types=num_gt, gate_embed_dim=32)
    model.load_state_dict(ckpt); model.eval()
    # Use the project's cache for dataset
    cache_dirs = gb_mod.glob(os.path.join(PROJ, 'cache*'))
    cache_dir = cache_dirs[0] if cache_dirs else 'cache_eval'
    ds_test = DelayDataset(static_p, dynamic_p, test_ids, scaler=None, cache_dir=cache_dir)
    loader = DataLoader(ds_test, batch_size=80, shuffle=False)
    preds = []; targets = []
    with torch.no_grad():
        for data in loader:
            out, _ = model(data.x, data.edge_index, data.batch, data.corner_cond, data.circuit_sig, getattr(data, 'struct_prior', None))
            preds.append(out.cpu().numpy()); targets.append(data.y.cpu().numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    mn = min(len(preds), len(test_dyn)); preds = preds[:mn]; targets = targets[:mn]; tdyn = test_dyn.iloc[:mn]
    rk = ranking_metrics(tdyn, preds, targets)
    hi = rk.get('hi_spread', {})
    pa = rk['pair_acc']
    results.append({
        'epoch': label, 'sp': rk['spearman'], 'regret': rk['regret_pct'],
        'top1': rk['top1_acc'], 'capture': rk['captured_pct'],
        'hi_sp': hi.get('spearman', float('nan')), 'hi_regret': hi.get('regret_pct', float('nan')),
        'hi_top1': hi.get('top1_acc', 0), 'hi_capture': hi.get('captured_pct', float('nan')),
        'pair_2': pa['<2%'][0], 'pair_5': pa['2-5%'][0], 'pair_10': pa['5-10%'][0], 'pair_20': pa['>10%'][0],
        'median_rel': float(np.median(np.abs(preds-targets)/targets)*100),
        'mae_ps': float(np.mean(np.abs(preds-targets))*1e12),
    })
    print(f"  ep{str(label):>5s}: hi_regret={results[-1]['hi_regret']:.2f}% hi_sp={results[-1]['hi_sp']:.3f} hi_top1={results[-1]['hi_top1']*100:.1f}% hi_cap={results[-1]['hi_capture']:.1f}%")

# Sort by high-spread regret (primary KPI)
# Sort by regret; within 0.3pp tiebreak by Spearman
results.sort(key=lambda r: r['hi_regret'] if not np.isnan(r['hi_regret']) else float('inf'))
best_reg = results[0]['hi_regret']
close = [r for r in results if r['hi_regret'] <= best_reg + 0.3]
best = max(close, key=lambda r: r['hi_sp'] if not np.isnan(r['hi_sp']) else -1.0)
print(f"\n=== Best Epoch Selection (regret primary, Spearman tiebreak <=0.3pp) ===")
print(f"{'epoch':>6s} {'hi_regret':>9s} {'hi_sp':>7s} {'hi_top1':>7s} {'hi_cap':>7s} {'sp':>6s} {'top1':>6s} {'cap':>6s} {'pair<2':>6s} {'pair2-5':>6s} {'pair5-10':>7s} {'pair>10':>7s} {'medrel':>7s} {'mae_ps':>7s}")
for r in results:
    marker = " <--BEST" if r is best else ""
    print(f"{str(r['epoch']):>6s} {r['hi_regret']:>8.2f}% {r['hi_sp']:>7.3f} {r['hi_top1']*100:>6.1f}% {r['hi_capture']:>6.1f}% {r['sp']:>6.3f} {r['top1']*100:>6.1f}% {r['capture']:>6.1f}% {r['pair_2']:>5.0f}% {r['pair_5']:>6.0f}% {r['pair_10']:>7.0f}% {r['pair_20']:>7.0f}% {r['median_rel']:>6.1f}% {r['mae_ps']:>6.1f}%{marker}")

print(f"\nBest epoch: {best['epoch']} | hi_regret={best['hi_regret']:.2f}% hi_sp={best['hi_sp']:.3f} hi_top1={best['hi_top1']*100:.1f}% hi_cap={best['hi_capture']:.1f}%")
print(f"pairwise: <2%={best['pair_2']:.0f}% 2-5%={best['pair_5']:.0f}% 5-10%={best['pair_10']:.0f}% >10%={best['pair_20']:.0f}%")
