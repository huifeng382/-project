"""基线对比评估：随机 / 晶体管数(TC) / 逻辑努力(LE) vs GNN(nowave集成 / wave)。

跑在 server（npz 在 ~/project-107-* 目录），在现有 V2 test 上离线对比，不重训。
方法：对每个候选算一个「延迟代理」，组内排序 -> regret/Spearman/top1/recall@2（avg_delay 口径）。

用法（server 端，项目根目录）：
  ~/venv/bin/python3 scripts/diag/_baseline_eval.py
"""
import sys, os, json
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
import numpy as np
import pandas as pd
import config
config.USE_V2 = True
config.USE_TRANSISTOR_WAVE = False
config.USE_SUPPLY_NOISE = False
config.USE_PARASITIC_CAPS = False
config.STRUCT_MODE = 'logic_only'

from src.utils import split_by_expr, ranking_metrics
from src.graph_builder import build_static_graph, rebuild_gate_types

HOME = os.path.expanduser('~')
NOWAVE_DIRS = ['v2nowave42', 'v2nowave123', 'v2nowave456',
               'v2nowave2468', 'v2nowave1357', 'v2nowave2024']
WAVE_DIRS = ['v2wave42', 'v2wave123']
RANDOM_TRIALS = 20
SPLIT = 42

# ---------- 加载 V2 test（与 _ens_struct.py 完全一致的行序） ----------
static_p = []; dynamic_p = []
for batch in ['batch_v2_full', 'batch_v2_io']:
    sp = os.path.join(PROJ, f'data/{batch}/circuit_static.parquet')
    dp = os.path.join(PROJ, f'data/{batch}/timing_arcs.parquet')
    if os.path.exists(sp) and os.path.exists(dp):
        static_p.append(sp); dynamic_p.append(dp)
    else:
        print(f"[WARN] 缺数据: {batch}（需在项目根目录跑）")
st = pd.concat([pd.read_parquet(p) for p in static_p], ignore_index=True)
for c in ['candidate', 'candidate_id']:
    if c in st.columns: st = st.rename(columns={c: 'circuit_id'})
st['circuit_id'] = st['circuit_id'].astype(str)
dyn = pd.concat([pd.read_parquet(p) for p in dynamic_p], ignore_index=True)
dyn = dyn.dropna(subset=['circuit_id', 'DELAY']); dyn['circuit_id'] = dyn['circuit_id'].astype(str)
dyn = dyn[(dyn['DELAY'] > 1e-12) & (dyn['DELAY'] < 1e-8)]
if 'expr' not in dyn.columns:
    dyn['expr'] = dyn['circuit_id'].map(dict(zip(st['circuit_id'], st['expr'].astype(str))))
ids = dyn['circuit_id'].unique().tolist()
id2e = dict(zip(dyn['circuit_id'].astype(str), dyn['expr'].astype(str)))
_, _, test_ids = split_by_expr(ids, id2e, seed=SPLIT)
test_dyn = dyn[dyn['circuit_id'].isin(test_ids)].reset_index(drop=True)
test_st = st[st['circuit_id'].isin(test_ids)].set_index('circuit_id')
print(f"test_dyn={len(test_dyn)} 行, 电路={test_dyn['circuit_id'].nunique()}, "
      f"组(>=2)={test_dyn.groupby('expr')['circuit_id'].nunique().gt(1).sum()}\n")

# ---------- 加载 npz：preds/targets 对齐到 test_dyn.iloc[:mn] ----------
def load_preds(dirs):
    arrs = []; tg = None
    for d in dirs:
        p = os.path.join(HOME, f'project-107-{d}/outputs/test_predictions.npz')
        if os.path.exists(p):
            data = np.load(p)
            arrs.append(data['preds'])
            tg = data['targets'] if tg is None else tg
        else:
            print(f"[WARN] 缺 {d}/test_predictions.npz")
    return (np.mean(arrs, axis=0) if arrs else None), tg

p_nowave, t_nowave = load_preds(NOWAVE_DIRS)
p_wave, _ = load_preds(WAVE_DIRS)
# 用 npz 自带的 targets 保证行序对齐
if t_nowave is not None:
    targets = t_nowave
else:
    targets = test_dyn['DELAY'].to_numpy(dtype=np.float64)
mn = min(len(test_dyn), len(targets)) if t_nowave is not None else len(test_dyn)
tdyn = test_dyn.iloc[:mn]; targets = targets[:mn]
assert len(tdyn) == len(targets), (len(tdyn), len(targets))

# 每电路标量 -> 广播成 per-row
def per_row(series):
    return np.array([float(series.get(c, float('nan'))) for c in tdyn['circuit_id']], dtype=np.float64)

def eval_method(name, pred):
    rk = ranking_metrics(tdyn, np.asarray(pred, dtype=np.float64), targets, avg_delay=True)
    hi = rk.get('hi_spread', {})
    g2 = hi.get('recall_at_k', {}).get(2, {}).get('strict', {}).get('hit_pct', float('nan'))
    pa = rk['pair_acc']
    print(f"  {name:20s} 全局遗憾={rk['regret_pct']:6.2f}%  hi遗憾={hi.get('regret_pct',float('nan')):6.2f}%  "
          f"Sp(hi)={hi.get('spearman',float('nan')):.3f}  top1(hi)={hi.get('top1_acc',0)*100:4.1f}%  "
          f"recall@2A(hi)={g2*100:4.1f}%  >10%成对={pa['>10%'][0]:.0f}%")
    return rk

print("===== 各方法（组内排序，avg_delay 口径；行数=%d）=====" % mn)
if p_nowave is not None:
    eval_method('GNN nowave(6seed)', p_nowave[:mn])
if p_wave is not None:
    eval_method('GNN wave(2seed)', p_wave[:mn])

# --- 随机基线：逐行均匀随机延迟，多试平均 ---
regs = []
for t in range(RANDOM_TRIALS):
    rng = np.random.RandomState(1000 + t)
    pred = rng.uniform(targets.min(), targets.max(), size=len(targets))
    regs.append(eval_method('', pred)['regret_pct'])
print(f"  {'Random(20试平均)':20s} 全局遗憾={np.mean(regs):6.2f}%\n")

# --- TC 基线：少晶体管 -> 快（rank by -transistor_count）---
eval_method('TC(晶体管数)', per_row(-test_st['transistor_count']))

# --- LE 代理：Σ over 门节点 (g*h + p)，用 build_static_graph 已算的 p/g/h ---
_nl_cache = {}
def circuit_le_proxy(cid):
    if cid in _nl_cache:
        return _nl_cache[cid]
    srow = test_st.loc[cid]
    nl = srow['gate_level_netlist']
    try:
        ip = json.loads(srow['input_pins_json']); op = json.loads(srow['output_pins_json'])
    except Exception:
        ip, op = [], []
    cells = set()
    for line in (nl or '').split('\n'):
        s = line.strip()
        if s.startswith('X_') and len(s.split()) >= 3:
            cells.add(s.split()[-1])
    rebuild_gate_types(cells)
    node_names, node_static, _ = build_static_graph('x', nl, ip or None, op or None)
    le = 0.0
    for i, n in enumerate(node_names):
        if not n.startswith('X_'):
            continue
        p = float(node_static[i, 4]); g = float(node_static[i, 5])
        h = float(np.expm1(node_static[i, 6]))   # h 存的是 log1p，还原
        le += g * max(h, 0.0) + p
    _nl_cache[cid] = le
    return le

le_by_circuit = {c: circuit_le_proxy(c) for c in test_st.index}
eval_method('LE(逻辑努力代理)', per_row(le_by_circuit))
