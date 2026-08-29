"""16.4.0 内存修复 sanity 验证
对比 DelayDataset 两条路径（重读 parquet vs 传入已过滤子集）：
- 行集 / 行序 / DELAY 逐位一致
- __getitem__ 输出（node_static / edge_index / y）逐位一致
用法: python scripts/diag/_sanity_memfix.py
"""
import sys
import os
import glob
import time
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import pandas as pd
import torch
from config import DATA_BATCHES, MIN_GROUP_SIZE, SPLIT_SEED
from src.utils import split_by_expr
from src.data_loader import DelayDataset

data_dir = ROOT
static_parquets, dynamic_parquets = [], []
for batch in DATA_BATCHES.split(','):
    batch = batch.strip()
    sp = os.path.join(data_dir, f"data/{batch}/circuit_static.parquet")
    dp = os.path.join(data_dir, f"data/{batch}/timing_arcs.parquet")
    if os.path.exists(sp) and os.path.exists(dp):
        static_parquets.append(sp)
        dynamic_parquets.append(dp)
    else:
        static_parquets.extend(sorted(glob.glob(os.path.join(data_dir, f"data/{batch}/circuit_static_part*.parquet"))))
        dynamic_parquets.extend(sorted(glob.glob(os.path.join(data_dir, f"data/{batch}/timing_arcs_part*.parquet"))))

t0 = time.time()

# ---- 复刻 main() 的预处理（train_sweep.py:278-340）----
dynamic_dfs = [pd.read_parquet(p) for p in dynamic_parquets]
dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)
if 'candidate_id' in dynamic_df.columns:
    dynamic_df = dynamic_df.rename(columns={'candidate_id': 'circuit_id'})
if 'delay_s' in dynamic_df.columns:
    if 'DELAY' not in dynamic_df.columns:
        dynamic_df = dynamic_df.rename(columns={'delay_s': 'DELAY'})
    else:
        dynamic_df['DELAY'] = dynamic_df['DELAY'].fillna(dynamic_df['delay_s'])
        dynamic_df = dynamic_df.drop(columns=['delay_s'])
dynamic_df = dynamic_df.dropna(subset=['circuit_id', 'DELAY'])
dynamic_df['circuit_id'] = dynamic_df['circuit_id'].astype(str)
dynamic_df = dynamic_df[(dynamic_df['DELAY'] > 1e-12) & (dynamic_df['DELAY'] < 1e-8)]
if MIN_GROUP_SIZE > 1 and 'expr' in dynamic_df.columns:
    gsize = dynamic_df.groupby('expr')['circuit_id'].nunique()
    keep_exprs = gsize[gsize >= MIN_GROUP_SIZE].index.astype(str)
    dynamic_df = dynamic_df[dynamic_df['expr'].astype(str).isin(keep_exprs)]
circuit_ids = dynamic_df['circuit_id'].unique().tolist()
id_to_expr = (dict(zip(dynamic_df['circuit_id'].astype(str), dynamic_df['expr'].astype(str)))
              if 'expr' in dynamic_df.columns else None)
train_ids, val_ids, test_ids = split_by_expr(circuit_ids, id_to_expr, seed=SPLIT_SEED)
print(f"split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} circuits")

# ---- 1) 小样本双路径 __getitem__ 对比（60 电路，构建快）----
sub_ids = train_ids[:60]
train_df = dynamic_df[dynamic_df['circuit_id'].isin(sub_ids)].reset_index(drop=True)
cache_old = os.path.join(ROOT, '_sanity_cache_old')
cache_new = os.path.join(ROOT, '_sanity_cache_new')
ds_old = DelayDataset(static_parquets, dynamic_parquets, sub_ids, scaler=None, cache_dir=cache_old)
ds_new = DelayDataset(static_parquets, dynamic_parquets, sub_ids, scaler=None, cache_dir=cache_new,
                      dynamic_df=train_df, prefiltered=True)
assert len(ds_old) == len(ds_new), f"len mismatch {len(ds_old)} vs {len(ds_new)}"
cids_old = ds_old.dynamic_df['circuit_id'].tolist()
cids_new = ds_new.dynamic_df['circuit_id'].tolist()
assert cids_old == cids_new, "circuit_id 顺序不一致"
print(f"[1] 小样本行集/行序一致: {len(ds_old)} 行")
for idx in [0, 1, 17, 59, len(ds_old) - 1]:
    a, b = ds_old[idx], ds_new[idx]
    assert torch.equal(a.x, b.x), f"idx={idx} x 不一致"
    assert torch.equal(a.edge_index, b.edge_index), f"idx={idx} edge_index 不一致"
    assert a.y == b.y, f"idx={idx} y 不一致"
    assert a.switching_pin == b.switching_pin, f"idx={idx} switching_pin 不一致"
    assert torch.equal(a.corner_cond, b.corner_cond), f"idx={idx} corner_cond 不一致"
print("[1] __getitem__ 输出逐位一致 (x/edge_index/y/switching_pin/corner_cond) OK")

# ---- 2) 全量行集/行序/DELAY 对比（不构建图，只比 df 本身）----
print("全量子集行序对比...")
tr_full = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)].reset_index(drop=True)
dfs = []
for p in dynamic_parquets:
    d = pd.read_parquet(p)
    d = d[d['DELAY'] > 1e-12]
    if 'candidate_id' in d.columns:
        d = d.rename(columns={'candidate_id': 'circuit_id'})
    d['circuit_id'] = d['circuit_id'].astype(str)
    dfs.append(d)
old_full = pd.concat(dfs, ignore_index=True)
old_tr = old_full[old_full['circuit_id'].isin(train_ids)].dropna(subset=['DELAY']).reset_index(drop=True)
assert len(old_tr) == len(tr_full), f"全量行数不一致 {len(old_tr)} vs {len(tr_full)}"
assert old_tr['circuit_id'].tolist() == tr_full['circuit_id'].tolist(), "全量行序不一致"
assert old_tr['DELAY'].tolist() == tr_full['DELAY'].tolist(), "全量 DELAY 不一致"
print(f"[2] 全量行集/行序/DELAY 一致: {len(tr_full)} 行 OK")

# ---- 3) wave OFF 时剔除列后的子集行数一致（列剔除不影响行）----
if 'transistor_wave_json' in dynamic_df.columns:
    wdf = dynamic_df.drop(columns=['transistor_wave_json'])
    assert len(wdf) == len(dynamic_df), "wave 列剔除影响行数"
    print("[3] wave 列剔除不影响行数 OK")

shutil.rmtree(cache_old, ignore_errors=True)
shutil.rmtree(cache_new, ignore_errors=True)
print(f"SANITY OK  耗时 {time.time() - t0:.0f}s")
