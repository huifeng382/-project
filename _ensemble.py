"""
4-seed ensemble evaluation (等权平均预测 → 排序指标)
Run on server from any seed directory: python _ensemble.py
"""
import numpy as np, sys, os
sys.path.insert(0, '.')
from src.utils import ranking_metrics

# Load predictions from all 4 seeds
seeds = ['rank','seed123','seed2024','seed456']
all_preds = []
for d in seeds:
    data = np.load(f'../project-107-{d}/outputs/test_predictions.npz')
    all_preds.append(data['preds'])
    print(f"Loaded {d}: {len(data['preds'])} preds")
ensemble_preds = np.mean(all_preds, axis=0)
targets = data['targets']

# Point accuracy (ensemble)
rel = np.abs(ensemble_preds - targets) / targets
print(f"\n=== Ensemble Point Accuracy ===")
print(f"Median Rel Err: {np.median(rel)*100:.2f}%")
print(f"Mean Abs Err: {np.mean(np.abs(ensemble_preds - targets))*1e12:.2f} ps")
print(f"Mean Rel Err: {np.mean(rel)*100:.2f}% (注: 被小延迟放大, 仅参考)")

# Ranking (need test_dyn — use current directory's dataset)
# Reconstruct test_dyn from the current run
import pandas as pd, json, glob as gb_module

# Find delivery parquet files (same as training uses)
static_p = []; dynamic_p = []
for prefix in ['data/delivery1','data/delivery2']:
    for b in ['batch1','batch2','batch3']:
        sp=f'{prefix}/{b}/circuit_static.parquet'
        dp=f'{prefix}/{b}/timing_arcs.parquet'
        if os.path.exists(sp) and os.path.exists(dp):
            static_p.append(sp); dynamic_p.append(dp)
            continue
        sparts=sorted(gb_module.glob(f'{prefix}/{b}/circuit_static_part*.parquet'))
        dparts=sorted(gb_module.glob(f'{prefix}/{b}/timing_arcs_part*.parquet'))
        if sparts and dparts:
            static_p.extend(sparts); dynamic_p.extend(dparts)

# Load and filter (replicate train_sweep pipeline)
dyn = pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn = dyn.dropna(subset=['circuit_id','DELAY'])
dyn['circuit_id']=dyn['circuit_id'].astype(str)
dyn = dyn[(dyn['DELAY']>1e-12)&(dyn['DELAY']<1e-8)]
st = pd.concat([pd.read_parquet(p) for p in static_p])
for col in ['candidate','candidate_id']:
    if col in st.columns: st=st.rename(columns={col:'circuit_id'})
st['circuit_id']=st['circuit_id'].astype(str)
four = set()
for _,r in st.iterrows():
    try:
        pins=json.loads(r['input_pins_json']) if isinstance(r['input_pins_json'],str) else r['input_pins_json']
        if sorted(pins)==['a','b','c','d']: four.add(r['circuit_id'])
    except: pass
dyn = dyn[dyn['circuit_id'].isin(four)]
if 'expr' not in dyn.columns and 'expr' in st.columns:
    id2e=dict(zip(st['circuit_id'],st['expr'].astype(str)))
    dyn['expr']=dyn['circuit_id'].map(id2e)
from src.utils import split_by_expr
ids=dyn['circuit_id'].unique().tolist()
id2e = dict(zip(dyn['circuit_id'].astype(str), dyn['expr'].astype(str))) if 'expr' in dyn.columns else None
_, _, test_ids = split_by_expr(ids, id2e, seed=42)
test_dyn = dyn[dyn['circuit_id'].isin(test_ids)].reset_index(drop=True)

# Ensure alignment
assert len(test_dyn) == len(ensemble_preds), f"Mismatch: {len(test_dyn)} vs {len(ensemble_preds)}"
print(f"\ntest_dyn rows: {len(test_dyn)}, preds: {len(ensemble_preds)}")

# Ranking
rk = ranking_metrics(test_dyn, ensemble_preds, targets)
print(f"\n=== Ensemble Ranking (4-seed 等权平均) ===")
print(f"[排序] 组(>=2)={rk['n_groups']}  Spearman={rk['spearman']:.3f}(→1)  选择遗憾={rk['regret_pct']:.2f}%(→0)  top1={rk['top1_acc']*100:.1f}%(→100)  捕获率={rk['captured_pct']:.1f}%(→100)  变体差中位={rk['spread_pct']:.1f}%")
hi = rk.get('hi_spread', {})
if hi.get('n',0)>0:
    print(f"[排序 spread>10%] 组(>=2)={hi['n']}  Spearman={hi['spearman']:.3f}(→1)  选择遗憾={hi['regret_pct']:.2f}%(→0)  top1={hi['top1_acc']*100:.1f}%(→100)  捕获率={hi['captured_pct']:.1f}%(→100)")
pa = rk['pair_acc']
print("[成对分辨(按真实延迟差)] " + "  ".join(f"{lab}:{pa[lab][0]:.0f}%(n={pa[lab][1]})" for lab in ['<2%','2-5%','5-10%','>10%']))

# Compare vs single best seed
print(f"\n=== vs Single Best Seed (seed123) ===")
d123 = np.load('../project-107-seed123/outputs/test_predictions.npz')
rk_single = ranking_metrics(test_dyn, d123['preds'], d123['targets'])
print(f"Single Spearman={rk_single['spearman']:.3f} 遗憾={rk_single['regret_pct']:.2f}%  top1={rk_single['top1_acc']*100:.1f}%")
his = rk_single.get('hi_spread',{})
if his.get('n',0)>0:
    print(f"Single hi-spread Spearman={his['spearman']:.3f} 遗憾={his['regret_pct']:.2f}%  top1={his['top1_acc']*100:.1f}%")
