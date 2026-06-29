"""
从 dataset_sweep_20expr 中随机抽取 1w 条样本，写入 data/batch_1w/。

用法:
    python src/sample_data.py                  # 默认 10000 条, seed=42
    python src/sample_data.py --n 5000         # 抽 5000 条
    python src/sample_data.py --seed 123       # 自定义随机种子
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import pandas as pd
import numpy as np

from src.utils import set_seed, create_dir


def main():
    parser = argparse.ArgumentParser(description="从 sweep 数据中随机抽取子集")
    parser.add_argument("--n", type=int, default=10000, help="抽取样本数 (默认 10000)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(data_dir, "data", "dataset_sweep_20expr")
    out_dir = os.path.join(data_dir, "data", "batch_1w")

    static_path = os.path.join(src_dir, "circuit_static.parquet")
    timing_path = os.path.join(src_dir, "timing_arcs.parquet")

    if not os.path.exists(static_path) or not os.path.exists(timing_path):
        raise FileNotFoundError(f"源数据不存在: {src_dir}")

    # ---------- 读取 ----------
    static_df = pd.read_parquet(static_path)
    timing_df = pd.read_parquet(timing_path)

    print(f"原始: static={len(static_df)} 电路, timing={len(timing_df)} 条")

    # ---------- 列名规范化 ----------
    for col in ['candidate', 'candidate_id']:
        if col in timing_df.columns and 'circuit_id' not in timing_df.columns:
            timing_df = timing_df.rename(columns={col: 'circuit_id'})
    timing_df['circuit_id'] = timing_df['circuit_id'].astype(str)

    if 'DELAY' not in timing_df.columns:
        for col in ['delay_s', 'delay']:
            if col in timing_df.columns:
                timing_df = timing_df.rename(columns={col: 'DELAY'})
                break

    for col in ['candidate', 'candidate_id']:
        if col in static_df.columns:
            static_df = static_df.rename(columns={col: 'circuit_id'})
    static_df['circuit_id'] = static_df['circuit_id'].astype(str)

    # ---------- 清洗 ----------
    before = len(timing_df)
    timing_df = timing_df.dropna(subset=['circuit_id', 'DELAY'])
    timing_df = timing_df[(timing_df['DELAY'] > 1e-12) & (timing_df['DELAY'] < 1e-8)]
    print(f"清洗: {before} → {len(timing_df)} 条 (剔除 {before - len(timing_df)})")

    # ---------- 随机抽取 ----------
    n_sample = min(args.n, len(timing_df))
    sampled_timing = timing_df.sample(n=n_sample, random_state=args.seed)
    print(f"随机抽取: {len(sampled_timing)} 条")

    # 筛选对应电路的静态数据
    sampled_circuits = sampled_timing['circuit_id'].unique()
    sampled_static = static_df[static_df['circuit_id'].isin(sampled_circuits)]

    print(f"涉及电路: {len(sampled_circuits)} 个")
    print(f"每条电路平均样本: {len(sampled_timing) / len(sampled_circuits):.1f}")

    # ---------- 写入 ----------
    create_dir(out_dir)
    sampled_static.to_parquet(os.path.join(out_dir, "circuit_static.parquet"), index=False)
    sampled_timing.to_parquet(os.path.join(out_dir, "timing_arcs.parquet"), index=False)

    # 元数据
    metadata = {
        "dataset_name": "batch_1w",
        "description": f"Random {n_sample} samples drawn from dataset_sweep_20expr (seed={args.seed})",
        "source": "dataset_sweep_20expr",
        "n_candidates": int(len(sampled_circuits)),
        "n_timing_arcs": int(len(sampled_timing)),
        "sample_seed": args.seed,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n已写入 {out_dir}/")
    print(f"  circuit_static.parquet  ({len(sampled_static)} 行)")
    print(f"  timing_arcs.parquet     ({len(sampled_timing)} 行)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
