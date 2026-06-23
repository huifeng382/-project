import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, ks_2samp, spearmanr, pearsonr
from config import *
from src.utils import set_seed, split_by_circuit
from src.data_loader import DelayDataset

# 设置种子以便复现
set_seed(RANDOM_SEED)

print("=" * 60)
print("数据质量检查报告")
print("=" * 60)

# ---------- 动态数据列名规范化函数（与 data_loader 保持一致） ----------
def normalize_dynamic(df):
    if 'circuit_id' not in df.columns:
        if 'candidate' in df.columns:
            df = df.rename(columns={'candidate': 'circuit_id'})
        elif 'candidate_id' in df.columns:
            df = df.rename(columns={'candidate_id': 'circuit_id'})
        else:
            raise KeyError(f"Dynamic data missing id column. Columns: {df.columns.tolist()}")
    df['circuit_id'] = df['circuit_id'].astype(str)
    if 'DELAY' not in df.columns:
        for col in ['delay', 'delay_s', 'Delay', 'delays']:
            if col in df.columns:
                df = df.rename(columns={col: 'DELAY'})
                break
    return df

# 1. 加载数据（与 train.py 一致）
static_parquets = glob.glob("data/batch_*/circuit_static.parquet")
dynamic_parquets = glob.glob("data/batch_*/timing_arcs.parquet")
if not static_parquets or not dynamic_parquets:
    raise FileNotFoundError("No Parquet files found.")

dynamic_dfs = []
for p in dynamic_parquets:
    df = pd.read_parquet(p)
    df = normalize_dynamic(df)   # 关键修复：规范化列名
    dynamic_dfs.append(df)
dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)
dynamic_df = dynamic_df.dropna(subset=['circuit_id', 'DELAY'])
# 注意：此时 DELAY 列已经存在，无需再 astype
dynamic_df = dynamic_df[(dynamic_df['DELAY'] > 1e-12) & (dynamic_df['DELAY'] < 1e-8)]

circuit_ids = dynamic_df['circuit_id'].unique().tolist()
train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)

# 标准化器（仅用于加载数据集，不影响检查）
train_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)]
all_cont_features = []
pins = ['a', 'b', 'c', 'd', 'e']
for _, row in train_dynamic.iterrows():
    for pin in pins:
        all_cont_features.append([row[f'slew_{pin}'], row[f'arrival_{pin}'], row[f'load_{pin}']])
scaler = StandardScaler(with_std=False)
scaler.fit(all_cont_features)

train_dataset = DelayDataset(static_parquets, dynamic_parquets, train_ids, scaler, CACHE_DIR)
val_dataset = DelayDataset(static_parquets, dynamic_parquets, val_ids, scaler, CACHE_DIR)
test_dataset = DelayDataset(static_parquets, dynamic_parquets, test_ids, scaler, CACHE_DIR)

def extract_all(dataset):
    X, y = [], []
    for i in range(len(dataset)):
        feat, label = dataset.extract_features(i)
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)

print("\n正在提取特征...")
X_train, y_train = extract_all(train_dataset)
X_val, y_val = extract_all(val_dataset)
X_test, y_test = extract_all(test_dataset)
print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# ---- 1. 目标变量（DELAY）分析 ----
print("\n" + "=" * 60)
print("1. 目标变量 (DELAY) 分析")
print("=" * 60)

def describe_target(y, name):
    stats = {
        'count': len(y),
        'min': np.min(y),
        'max': np.max(y),
        'mean': np.mean(y),
        'median': np.median(y),
        'std': np.std(y),
        'percentiles_1': np.percentile(y, 1),
        'percentiles_5': np.percentile(y, 5),
        'percentiles_95': np.percentile(y, 95),
        'percentiles_99': np.percentile(y, 99),
        'skew': skew(y),
        'kurtosis': kurtosis(y),
    }
    print(f"\n{name} 统计:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")
    return stats

stats_train = describe_target(y_train, "训练集")
describe_target(y_val, "验证集")
describe_target(y_test, "测试集")

# 检查极端离群点（log10域）
print("\n检查极端离群点（log10域）:")
y_log_train = np.log10(y_train + 1e-12)
y_log_val = np.log10(y_val + 1e-12)
y_log_test = np.log10(y_test + 1e-12)
print(f"训练集 log10(DELAY) 范围: [{np.min(y_log_train):.2f}, {np.max(y_log_train):.2f}]")
print(f"验证集 log10(DELAY) 范围: [{np.min(y_log_val):.2f}, {np.max(y_log_val):.2f}]")
print(f"测试集 log10(DELAY) 范围: [{np.min(y_log_test):.2f}, {np.max(y_log_test):.2f}]")

# ---- 2. 分布一致性 ----
print("\n" + "=" * 60)
print("2. 训练/验证/测试目标分布一致性")
print("=" * 60)
ks_train_val = ks_2samp(y_log_train, y_log_val)
ks_train_test = ks_2samp(y_log_train, y_log_test)
print(f"训练集 vs 验证集 K-S 检验: p-value = {ks_train_val.pvalue:.4f}")
print(f"训练集 vs 测试集 K-S 检验: p-value = {ks_train_test.pvalue:.4f}")
if ks_train_val.pvalue < 0.05:
    print("  -> 警告: 训练集与验证集分布显著不同，可能导致泛化问题")
if ks_train_test.pvalue < 0.05:
    print("  -> 警告: 训练集与测试集分布显著不同，可能导致泛化问题")

# ---- 3. 电路划分检查 ----
print("\n" + "=" * 60)
print("3. 电路划分检查（应无重叠）")
print("=" * 60)
train_circuits = set(train_dataset.dynamic_df['circuit_id'])
val_circuits = set(val_dataset.dynamic_df['circuit_id'])
test_circuits = set(test_dataset.dynamic_df['circuit_id'])
print(f"训练电路数: {len(train_circuits)}")
print(f"验证电路数: {len(val_circuits)}")
print(f"测试电路数: {len(test_circuits)}")
overlap_train_val = len(train_circuits & val_circuits)
overlap_train_test = len(train_circuits & test_circuits)
overlap_val_test = len(val_circuits & test_circuits)
print(f"训练∩验证重叠: {overlap_train_val}")
print(f"训练∩测试重叠: {overlap_train_test}")
print(f"验证∩测试重叠: {overlap_val_test}")
if overlap_train_val + overlap_train_test + overlap_val_test > 0:
    print("  -> 警告: 存在电路重叠，数据划分可能不当")

# ---- 4. 特征分析 ----
print("\n" + "=" * 60)
print("4. 特征分析")
print("=" * 60)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_val_scaled = scaler_x.transform(X_val)
X_test_scaled = scaler_x.transform(X_test)

# 常数特征
variances = np.var(X_train_scaled, axis=0)
constant_idx = np.where(variances < 1e-12)[0]
print(f"常数特征数: {len(constant_idx)} / {X_train_scaled.shape[1]}")
if len(constant_idx) > 0:
    print(f"  常数特征索引: {constant_idx}")

print(f"训练集 NaN 数: {np.isnan(X_train_scaled).sum()}")
print(f"训练集 Inf 数: {np.isinf(X_train_scaled).sum()}")

# 特征与目标相关性
print("\n特征与目标 (log10 DELAY) 相关性分析:")
y_log = np.log10(y_train + 1e-12)
pearson_corr = []
spearman_corr = []
for i in range(X_train_scaled.shape[1]):
    # 跳过常数特征
    if variances[i] < 1e-12:
        pearson_corr.append(np.nan)
        spearman_corr.append(np.nan)
        continue
    p, _ = pearsonr(X_train_scaled[:, i], y_log)
    s, _ = spearmanr(X_train_scaled[:, i], y_log)
    pearson_corr.append(p)
    spearman_corr.append(s)

pearson_corr = np.array(pearson_corr)
spearman_corr = np.array(spearman_corr)

# 显示非 NaN 的前10大相关性
valid_pearson = np.where(~np.isnan(pearson_corr))[0]
if len(valid_pearson) > 0:
    top_pearson_idx = valid_pearson[np.argsort(np.abs(pearson_corr[valid_pearson]))[::-1][:10]]
    print("\nTop 10 特征（按皮尔逊相关系数绝对值）:")
    for idx in top_pearson_idx:
        print(f"  Feat {idx}: Pearson={pearson_corr[idx]:.4f}, Spearman={spearman_corr[idx]:.4f}")
else:
    print("  没有有效的非零方差特征来计算相关性。")

valid_spearman = np.where(~np.isnan(spearman_corr))[0]
if len(valid_spearman) > 0:
    top_spearman_idx = valid_spearman[np.argsort(np.abs(spearman_corr[valid_spearman]))[::-1][:10]]
    print("\nTop 10 特征（按斯皮尔曼相关系数绝对值）:")
    for idx in top_spearman_idx:
        print(f"  Feat {idx}: Spearman={spearman_corr[idx]:.4f}, Pearson={pearson_corr[idx]:.4f}")
else:
    print("  没有有效的非零方差特征来计算相关性。")

high_corr = np.where(np.abs(pearson_corr) > 0.9)[0]
if len(high_corr) > 0:
    print(f"  -> 警告: 存在与目标高度相关的特征 (Pearson > 0.9)，可能数据泄漏: {high_corr}")

# ---- 5. 特征分布差异 ----
print("\n" + "=" * 60)
print("5. 特征分布一致性（训练 vs 验证/测试）")
print("=" * 60)
mean_train = np.mean(X_train_scaled, axis=0)
mean_val = np.mean(X_val_scaled, axis=0)
mean_test = np.mean(X_test_scaled, axis=0)
diff_train_val = np.abs(mean_train - mean_val)
diff_train_test = np.abs(mean_train - mean_test)
print("训练集与验证集特征均值差异（前10大）:")
top_diff_idx = np.argsort(diff_train_val)[::-1][:10]
for idx in top_diff_idx:
    print(f"  Feat {idx}: diff = {diff_train_val[idx]:.4f}")
print("训练集与测试集特征均值差异（前10大）:")
top_diff_idx_test = np.argsort(diff_train_test)[::-1][:10]
for idx in top_diff_idx_test:
    print(f"  Feat {idx}: diff = {diff_train_test[idx]:.4f}")

# ---- 6. 总结与建议 ----
print("\n" + "=" * 60)
print("6. 总结与建议")
print("=" * 60)

issues = []
if ks_train_val.pvalue < 0.05 or ks_train_test.pvalue < 0.05:
    issues.append("目标分布不一致，建议重新随机划分或采用分层采样。")
if overlap_train_val + overlap_train_test + overlap_val_test > 0:
    issues.append("电路重叠，请确保 split_by_circuit 正确工作。")
if len(constant_idx) > 0:
    issues.append(f"存在 {len(constant_idx)} 个常数特征，应删除。")
if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
    issues.append("特征包含 NaN/Inf，需清洗。")
if len(high_corr) > 0:
    issues.append(f"存在与目标高度相关的特征（可能泄漏），需检查特征生成过程。")
if stats_train['skew'] > 2 or stats_train['kurtosis'] > 10:
    issues.append("目标变量分布严重偏态，变换（如 Box-Cox）可能不够，尝试更强变换。")
if np.max(y_train) / np.min(y_train) > 1e6:
    issues.append("目标变量动态范围极大，建议进一步截断或使用对数变换。")

if issues:
    print("发现以下潜在问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("未发现明显数据问题。模型误差可能源于特征表达能力不足或模型复杂度不够。")

print("\n检查完成。请根据以上报告针对性地优化数据或模型。")