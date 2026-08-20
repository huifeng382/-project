"""structrich / structlogic 多 seed 集成评估（server 端运行，不依赖 cwd）。

用法：
  # 把本文件拷到 server 任意目录，然后：
  cd ~/project-107-rank && ~/venv/bin/python3 _ens_struct.py

打印：
  1) 各单 seed 的 hi_spread 指标（npz 计算） + 与 SUMMARY 遗憾自动对账
     —— 若偏离 > 0.5pp 说明 npz 不是 midpoint 选点（14.1.1 的 bug），
        需先 EVAL_ONLY=midpoint 重生成 npz 再集成
  2) 全部 N_ENS-seed 组合（DIRS 内取 N_ENS 个，等权平均）
  3) 全量平均 + 按单 seed 遗憾排序的 trim 扫描（keep top-K）
"""
import numpy as np, os, pandas as pd, json, sys, glob as gb_mod, itertools
HOME = os.path.expanduser('~')
ROOT = os.path.join(HOME, 'project-107-rank')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
from src.utils import split_by_expr, ranking_metrics

# ---- 参与集成的 run（目录名 = ~/project-107-<dir>）----
# 'structlogic' = 7.3 批次的 seed42 run（dir 无 seed 后缀，TRAIN_SEED=42 默认）
DIRS = ['structlogic', 'structlogic2468', 'structlogic456']
N_ENS = 3  # 主推的集成 seed 数

# ---- SUMMARY 里的 hi_spread 遗憾（对账用；新增 run 就加一行）----
EXPECT_REGRET = {
    'structlogic': 3.64, 'structlogic2468': 1.72, 'structlogic456': 1.78,
}

# --- 加载 test 动态数据（与 _trim8.py 完全一致，保证可复现）---
static_p = []; dynamic_p = []
for prefix in ['data/delivery1', 'data/delivery2']:
    for b in ['batch1', 'batch2', 'batch3']:
        sp = f'{prefix}/{b}/circuit_static.parquet'; dp = f'{prefix}/{b}/timing_arcs.parquet'
        if os.path.exists(sp) and os.path.exists(dp):
            static_p.append(sp); dynamic_p.append(dp); continue
        for t in ['circuit_static', 'timing_arcs']:
            for p in sorted(gb_mod.glob(f'{prefix}/{b}/{t}_part*.parquet')):
                (static_p if 'static' in p else dynamic_p).append(p)
dyn_all = pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn_all = dyn_all.dropna(subset=['circuit_id', 'DELAY']); dyn_all['circuit_id'] = dyn_all['circuit_id'].astype(str)
dyn_all = dyn_all[(dyn_all['DELAY'] > 1e-12) & (dyn_all['DELAY'] < 1e-8)]
st = pd.concat([pd.read_parquet(p) for p in static_p])
for c in ['candidate', 'candidate_id']:
    if c in st.columns: st = st.rename(columns={c: 'circuit_id'})
st['circuit_id'] = st['circuit_id'].astype(str)
four = set()
for _, r in st.iterrows():
    try:
        pins = json.loads(r['input_pins_json']) if isinstance(r['input_pins_json'], str) else r['input_pins_json']
        if sorted(pins) == ['a', 'b', 'c', 'd']: four.add(r['circuit_id'])
    except: pass
dyn_all = dyn_all[dyn_all['circuit_id'].isin(four)]
if 'expr' not in dyn_all.columns:
    dyn_all['expr'] = dyn_all['circuit_id'].map(dict(zip(st['circuit_id'], st['expr'].astype(str))))
ids = dyn_all['circuit_id'].unique().tolist()
id2e = dict(zip(dyn_all['circuit_id'].astype(str), dyn_all['expr'].astype(str)))
_, _, test_ids = split_by_expr(ids, id2e, seed=42)
test_dyn = dyn_all[dyn_all['circuit_id'].isin(test_ids)].reset_index(drop=True)
print(f"test_dyn rows: {len(test_dyn)}\n")

# --- 加载各 run 的 npz + 单 seed 指标 + 对账 ---
preds_by_dir = {}; regret_by_dir = {}; targets = None; missing = []
for d in DIRS:
    p = os.path.join(HOME, f'project-107-{d}/outputs/test_predictions.npz')
    if not os.path.exists(p):
        missing.append(d); preds_by_dir[d] = None; regret_by_dir[d] = 1e9
        print(f"  {d}: [MISSING npz] -> {p}"); continue
    data = np.load(p)
    preds_by_dir[d] = data['preds']; targets = data['targets']
    mn = min(len(test_dyn), len(data['preds']))
    rk = ranking_metrics(test_dyn.iloc[:mn], data['preds'][:mn], data['targets'][:mn])
    hi = rk.get('hi_spread', {}); rk2 = hi.get('recall_at_k', {})
    reg = hi.get('regret_pct', 100)
    regret_by_dir[d] = reg
    r2A = rk2.get(2, {}).get('strict', {}).get('hit_pct', float('nan'))
    exp = EXPECT_REGRET.get(d)
    flag = '' if exp is None else ('' if abs(reg - exp) <= 0.5 else '  <-- npz 与 SUMMARY 不符! 需 EVAL_ONLY=midpoint 重生成')
    print(f"  {d}: regret={reg:.2f}% sp={hi.get('spearman',0):.3f} top1={hi.get('top1_acc',0)*100:.1f}% "
          f"cap={hi.get('captured_pct',0):.1f}% recall@2A={r2A*100:.1f}%  (SUMMARY {exp}%){flag}")
if missing:
    print(f"\n[WARN] missing npz: {missing} —— 先补跑/补拷，或从 DIRS 里去掉")
print()

def summarize(names, label):
    keep = [d for d in names if preds_by_dir[d] is not None]
    if not keep:
        print(f"{label}: no available seeds"); return
    ens = np.mean([preds_by_dir[d] for d in keep], axis=0)
    mn = min(len(test_dyn), len(ens)); tdyn = test_dyn.iloc[:mn]
    rk = ranking_metrics(tdyn, ens[:mn], targets[:mn])
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
    print(f"  seeds: {keep}  单 seed 遗憾: {[f'{regret_by_dir[d]:.2f}%' for d in keep]}")

# --- 主推：N_ENS 个 seed 的所有组合 ---
print(f"===== {N_ENS}-seed 组合（等权平均） =====")
for combo in itertools.combinations(DIRS, N_ENS):
    summarize(list(combo), ' + '.join(combo))

# --- 全量 + trim 扫描 ---
print(f"\n===== 全量 / trim 扫描 =====")
summarize(DIRS, f'ALL-{len(DIRS)} 平均')
order = sorted(DIRS, key=lambda d: regret_by_dir[d])
for K in range(len(DIRS), 1, -1):
    summarize(order[:K], f'trim keep top-{K} (去掉最差 {len(DIRS)-K} 个: {[d for d in order[K:]]})')
