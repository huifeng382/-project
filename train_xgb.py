import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from config import *
from src.utils import set_seed, split_by_circuit, create_dir
from src.data_loader import DelayDataset

set_seed(RANDOM_SEED)
create_dir(CACHE_DIR)
create_dir(OUTPUT_DIR)

static_parquets = glob.glob("data/batch_05/circuit_static.parquet")
dynamic_parquets = glob.glob("data/batch_05/timing_arcs.parquet")
if not static_parquets or not dynamic_parquets:
    raise FileNotFoundError("No Parquet files found.")

# 合并动态数据并清洗
dynamic_dfs = [pd.read_parquet(p) for p in dynamic_parquets]
dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)

# 列名规范化
for col in ['candidate', 'candidate_id']:
    if col in dynamic_df.columns and 'circuit_id' not in dynamic_df.columns:
        dynamic_df = dynamic_df.rename(columns={col: 'circuit_id'})
dynamic_df['circuit_id'] = dynamic_df['circuit_id'].astype(str)
if 'DELAY' not in dynamic_df.columns:
    for col in ['delay_s', 'delay']:
        if col in dynamic_df.columns:
            dynamic_df = dynamic_df.rename(columns={col: 'DELAY'})
            break

dynamic_df = dynamic_df.dropna(subset=['circuit_id', 'DELAY'])
dynamic_df = dynamic_df[(dynamic_df['DELAY'] > 1e-12) & (dynamic_df['DELAY'] < 1e-8)]

circuit_ids = dynamic_df['circuit_id'].unique().tolist()
train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)

# extract_features 不使用 scaler，传 None 即可
scaler = None

train_dataset = DelayDataset(static_parquets, dynamic_parquets, train_ids, scaler, CACHE_DIR)
val_dataset   = DelayDataset(static_parquets, dynamic_parquets, val_ids,   scaler, CACHE_DIR)
test_dataset  = DelayDataset(static_parquets, dynamic_parquets, test_ids,  scaler, CACHE_DIR)

def extract_all(dataset):
    X, y = [], []
    for i in range(len(dataset)):
        feat, label = dataset.extract_features(i)
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = extract_all(train_dataset)
X_val, y_val = extract_all(val_dataset)
X_test, y_test = extract_all(test_dataset)

# ---- 离群点清洗（基于 DELAY 分位数） ----
lower = np.percentile(y_train, 1)
upper = np.percentile(y_train, 99)
keep = (y_train > lower) & (y_train < upper)
X_train = X_train[keep]
y_train = y_train[keep]
print(f"离群点清洗后训练集大小: {len(X_train)}")
# --------------------------------------------

# ---- 特征标准化 ----
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

# ==================== 目标变量变换（log10，与 GNN 一致） ====================
y_train_target = np.log10(y_train + 1e-12)
y_val_target   = np.log10(y_val + 1e-12)
y_test_target  = np.log10(y_test + 1e-12)

def inverse_transform(preds_target):
    """逆变换回原始延迟"""
    preds = 10 ** preds_target
    preds = np.clip(preds, 1e-12, 1e-8)
    return preds

# ==================== 训练 LightGBM ====================
# 创建 LightGBM 数据集（支持早停和验证）
train_data = lgb.Dataset(X_train, label=y_train_target)
val_data = lgb.Dataset(X_val, label=y_val_target, reference=train_data)

# 参数配置（可调节）
params = {
    'objective': 'regression',           # 回归任务
    'metric': 'mae',                     # 评估指标（早停依据）
    'boosting_type': 'gbdt',             # 梯度提升树
    'num_leaves': 31,                    # 叶子节点数，控制复杂度
    'max_depth': 8,                      # 最大深度
    'learning_rate': 0.05,               # 学习率
    'feature_fraction': 0.8,             # 列采样
    'bagging_fraction': 0.8,             # 行采样
    'bagging_freq': 5,                   # 每5轮做一次bagging
    'reg_alpha': 0.1,                    # L1正则
    'reg_lambda': 0.1,                   # L2正则
    'min_child_samples': 20,             # 叶子最小样本数
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

# 训练模型（带早停）
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
)

# 预测
preds_val_target = model.predict(X_val, num_iteration=model.best_iteration)
preds_test_target = model.predict(X_test, num_iteration=model.best_iteration)

# 逆变换回原始延迟
preds_val = inverse_transform(preds_val_target)
preds_test = inverse_transform(preds_test_target)

# 计算相对误差
rel_err_val = np.abs(preds_val - y_val) / y_val * 100
rel_err_test = np.abs(preds_test - y_test) / y_test * 100

print(f"LightGBM Val Mean Relative Error: {np.mean(rel_err_val):.2f}%")
print(f"LightGBM Test Mean Relative Error: {np.mean(rel_err_test):.2f}%")
np.savez(os.path.join(OUTPUT_DIR, 'lgb_predictions.npz'), preds=preds_test, targets=y_test)

# Per-corner breakdown
test_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(test_ids)]
if 'corner' in test_dynamic.columns and len(preds_test) > 0:
    print("\nPer-corner relative error:")
    corners = test_dynamic['corner'].values
    if len(corners) == len(preds_test):
        for c in sorted(set(corners)):
            mask = corners == c
            if mask.sum() > 0:
                err = np.abs(preds_test[mask] - y_test[mask]) / y_test[mask] * 100
                print(f"  {c}: {np.mean(err):.2f}%")