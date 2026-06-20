import os
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path

# ========== 配置参数 ==========
SOURCE_DIR = "data/batch_02"          # 源数据目录
TARGET_DIR = "data/batch_03"          # 输出目录
TARGET_SAMPLES = 8000                 # 目标样本数（约 5k~1w 之间）
RANDOM_SEED = 42                      # 固定随机种子，确保可重复

# 源文件路径
static_path = os.path.join(SOURCE_DIR, "circuit_static.parquet")
dynamic_path = os.path.join(SOURCE_DIR, "timing_arcs.parquet")

# 检查文件是否存在
if not os.path.exists(static_path) or not os.path.exists(dynamic_path):
    raise FileNotFoundError(f"请确保 {SOURCE_DIR} 下存在 circuit_static.parquet 和 timing_arcs.parquet")

# ========== 读取数据 ==========
print("正在读取数据...")
static_df = pd.read_parquet(static_path)
dynamic_df = pd.read_parquet(dynamic_path)

# ========== 数据清洗 ==========
print("执行数据清洗...")
original_len = len(dynamic_df)
dynamic_df = dynamic_df.dropna(subset=['circuit_id'])
dynamic_df['circuit_id'] = dynamic_df['circuit_id'].astype(str)
dynamic_df = dynamic_df.dropna(subset=['DELAY'])
removed = original_len - len(dynamic_df)
if removed > 0:
    print(f"  清洗掉 {removed} 行（DELAY 或 circuit_id 为 NaN）")
print(f"  清洗后剩余 {len(dynamic_df)} 条样本")

# ========== 按电路采样 ==========
print("按电路进行随机采样...")
np.random.seed(RANDOM_SEED)

# 统计每个电路的样本数
circuit_counts = dynamic_df['circuit_id'].value_counts()
avg_samples = circuit_counts.mean()
num_circuits = int(np.ceil(TARGET_SAMPLES / avg_samples))
all_circuits = dynamic_df['circuit_id'].unique().tolist()

if num_circuits > len(all_circuits):
    num_circuits = len(all_circuits)
    print(f"  目标样本数过大，将使用全部 {num_circuits} 个电路")

# 随机选择电路
selected_circuits = np.random.choice(all_circuits, size=num_circuits, replace=False).tolist()

# 筛选动态数据
dynamic_sampled = dynamic_df[dynamic_df['circuit_id'].isin(selected_circuits)].reset_index(drop=True)
print(f"  选中 {num_circuits} 个电路，实际样本数：{len(dynamic_sampled)} (目标 {TARGET_SAMPLES})")
print(f"  偏差：{len(dynamic_sampled) - TARGET_SAMPLES:+d}")

# 筛选静态数据（只保留选中电路）
static_sampled = static_df[static_df['circuit_id'].isin(selected_circuits)].reset_index(drop=True)

# ========== 创建目标目录并保存 ==========
os.makedirs(TARGET_DIR, exist_ok=True)
static_out = os.path.join(TARGET_DIR, "circuit_static.parquet")
dynamic_out = os.path.join(TARGET_DIR, "timing_arcs.parquet")

static_sampled.to_parquet(static_out, index=False)
dynamic_sampled.to_parquet(dynamic_out, index=False)

print(f"\n采样完成！数据已保存到 {TARGET_DIR}")
print(f"  - circuit_static.parquet: {len(static_sampled)} 个电路")
print(f"  - timing_arcs.parquet:    {len(dynamic_sampled)} 条样本")