"""通用集成脚本：任意 seed 列表的等权平均（V2 数据，avg_delay 口径，默认不剪枝）。

用法（server 端，在项目目录跑）：
  ~/venv/bin/python3 scripts/diag/_ens_struct.py [seed1 seed2 ...] [--trim]

  - 不传 seed → 默认 no-wave 6-seed 全平均（当前交付推荐：等权、无 post-hoc 偏差）
  - --trim → 额外打印按 test regret 的剪枝扫描（仅供参考；主决策用全平均，见 PROJECT_LOG 15.2.3）

数据：V2 的 batch_v2_full + batch_v2_io（USE_V2 口径，expr 切分 SPLIT_SEED=42），
评估用 avg_delay=True（对齐 Rust，每电路均值）。
"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod, argparse

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
from src.utils import split_by_expr, ranking_metrics

HOME = os.path.expanduser('~')
DEFAULT_SEEDS = ['v2nowave42', 'v2nowave123', 'v2nowave456',
                 'v2nowave2468', 'v2nowave1357', 'v2nowave2024']

ap = argparse.ArgumentParser()
ap.add_argument('seeds', nargs='*', help='参与集成的 run 目录名（~/project-107-<name>）')
ap.add_argument('--trim', action='store_true', help='额外打印 test-regret 剪枝扫描（参考）')
args = ap.parse_args()
DIRS = args.seeds or DEFAULT_SEEDS

# --- 加载 V2 test 动态数据（batch_v2_full + batch_v2_io，expr 切分 seed 42）---
static_p = []; dynamic_p = []
for batch in ['batch_v2_full', 'batch_v2_io']:
    sp = os.path.join(PROJ, f'data/{batch}/circuit_static.parquet')
    dp = os.path.join(PROJ, f'data/{batch}/timing_arcs.parquet')
    if os.path.exists(sp) and os.path.exists(dp):
        static_p.append(sp); dynamic_p.append(dp)
    else:
        print(f"[WARN] 缺数据: {batch}")
dyn_all = pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn_all = dyn_all.dropna(subset=['circuit_id', 'DELAY']); dyn_all['circuit_id'] = dyn_all['circuit_id'].astype(str)
dyn_all = dyn_all[(dyn_all['DELAY'] > 1e-12) & (dyn_all['DELAY'] < 1e-8)]
st = pd.concat([pd.read_parquet(p) for p in static_p])
for c in ['candidate', 'candidate_id']:
    if c in st.columns: st = st.rename(columns={c: 'circuit_id'})
st['circuit_id'] = st['circuit_id'].astype(str)
if 'expr' not in dyn_all.columns:
    dyn_all['expr'] = dyn_all['circuit_id'].map(dict(zip(st['circuit_id'], st['expr'].astype(str))))
ids = dyn_all['circuit_id'].unique().tolist()
id2e = dict(zip(dyn_all['circuit_id'].astype(str), dyn_all['expr'].astype(str)))
_, _, test_ids = split_by_expr(ids, id2e, seed=42)
test_dyn = dyn_all[dyn_all['circuit_id'].isin(test_ids)].reset_index(drop=True)
print(f"test_dyn rows: {len(test_dyn)}  组(>=2): {test_dyn.groupby('expr')['circuit_id'].nunique().gt(1).sum()}\n")

# --- 加载各 run 的 npz + 单 seed 指标 ---
preds_by_dir = {}; regret_by_dir = {}; targets = None; missing = []
for d in DIRS:
    p = os.path.join(HOME, f'project-107-{d}/outputs/test_predictions.npz')
    if not os.path.exists(p):
        missing.append(d); preds_by_dir[d] = None; regret_by_dir[d] = 1e9
        print(f"  {d}: [MISSING npz] -> {p}"); continue
    data = np.load(p)
    preds_by_dir[d] = data['preds']; targets = data['targets']
    mn = min(len(test_dyn), len(data['preds']))
    hi = ranking_metrics(test_dyn.iloc[:mn], data['preds'][:mn], data['targets'][:mn],
                         avg_delay=True).get('hi_spread', {})
    reg = hi.get('regret_pct', 100); regret_by_dir[d] = reg
    print(f"  {d}: hi_regret={reg:.2f}% sp={hi.get('spearman',0):.3f} "
          f"top1={hi.get('top1_acc',0)*100:.1f}% cap={hi.get('captured_pct',0):.1f}%")
if missing:
    print(f"\n[WARN] missing npz: {missing} —— 补跑/补拷，或从参数里去掉")
print()

def summarize(names, label):
    keep = [d for d in names if preds_by_dir[d] is not None]
    if not keep:
        print(f"{label}: no available seeds"); return
    ens = np.mean([preds_by_dir[d] for d in keep], axis=0)
    mn = min(len(test_dyn), len(ens)); tdyn = test_dyn.iloc[:mn]
    rk = ranking_metrics(tdyn, ens[:mn], targets[:mn], avg_delay=True)
    hi = rk.get('hi_spread', {}); rk2 = hi.get('recall_at_k', {}); pa = rk['pair_acc']
    g2 = rk2.get(2, {}); g3 = rk2.get(3, {})
    print(f"\n=== {label} ===")
    print(f"  全局:   Spearman={rk['spearman']:.3f}  遗憾={rk['regret_pct']:.2f}%  top1={rk['top1_acc']*100:.1f}%  "
          f"捕获={rk['captured_pct']:.1f}%  组数={rk['n_groups']}")
    print(f"  hi_spread:  Spearman={hi.get('spearman',0):.3f}  遗憾={hi.get('regret_pct',0):.2f}%  "
          f"top1={hi.get('top1_acc',0)*100:.1f}%  捕获={hi.get('captured_pct',0):.1f}%  组数={hi.get('n',0)}")
    print(f"  recall@K(hi): @2 A={g2.get('strict',{}).get('hit_pct',float('nan'))*100:.1f}% "
          f"B={g2.get('lenient',{}).get('hit_pct',float('nan'))*100:.1f}% "
          f"(n={g2.get('strict',{}).get('n',0)})  "
          f"@3 A={g3.get('strict',{}).get('hit_pct',float('nan'))*100:.1f}% "
          f"B={g3.get('lenient',{}).get('hit_pct',float('nan'))*100:.1f}% "
          f"(n={g3.get('strict',{}).get('n',0)})")
    print(f"  成对: <2%={pa['<2%'][0]:.0f}%  2-5%={pa['2-5%'][0]:.0f}%  5-10%={pa['5-10%'][0]:.0f}%  >10%={pa['>10%'][0]:.0f}%")
    print(f"  seeds: {keep}")

# --- 主推：全量等权平均（无 post-hoc 剪枝，见 15.2.3 决策）---
print("===== 主推：全量等权平均（不剪枝） =====")
summarize(DIRS, f'ALL-{len(DIRS)} 等权平均')

# --- 参考：单 seed 遗憾排序 + 剪枝扫描 ---
if args.trim:
    print("\n===== 参考：test-regret 剪枝扫描（post-hoc，仅参考） =====")
    order = sorted([d for d in DIRS if preds_by_dir[d] is not None], key=lambda d: regret_by_dir[d])
    for K in range(len(order), 1, -1):
        summarize(order[:K], f'trim keep top-{K} (去掉: {[d for d in order[K:]]})')
