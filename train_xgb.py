import sys
import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from config import *
from src.utils import set_seed, split_by_circuit, create_dir
from src.data_loader import DelayDataset

# ========== 新增导入 ==========
from scipy.stats import boxcox
from scipy.special import inv_boxcox
# =============================

set_seed(RANDOM_SEED)
create_dir(OUTPUT_DIR)

static_parquets = glob.glob("data/batch_*/circuit_static.parquet")
dynamic_parquets = glob.glob("data/batch_*/timing_arcs.parquet")
if not static_parquets or not dynamic_parquets:
    raise FileNotFoundError("No Parquet files found.")

# 合并动态数据并清洗
dynamic_dfs = [pd.read_parquet(p) for p in dynamic_parquets]
dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)
dynamic_df = dynamic_df.dropna(subset=['circuit_id', 'DELAY'])
dynamic_df['circuit_id'] = dynamic_df['circuit_id'].astype(str)
dynamic_df = dynamic_df[(dynamic_df['DELAY'] > 1e-12) & (dynamic_df['DELAY'] < 1e-8)]

circuit_ids = dynamic_df['circuit_id'].unique().tolist()
train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)

# 标准化器（与 train.py 一致）
train_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)]
all_cont_features = []
pins = ['a','b','c','d','e']
for _, row in train_dynamic.iterrows():
    for pin in pins:
        all_cont_features.append([row[f'slew_{pin}'], row[f'arrival_{pin}'], row[f'load_{pin}']])
scaler = StandardScaler(with_std=False)
scaler.fit(all_cont_features)

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

# ==================== 目标变量变换选项 ====================
# 选择一种变换方式（取消注释对应的代码块，注释掉其他）

# ---------- 选项 0: 原始 log10（基线） ----------
# y_train_target = np.log10(y_train + 1e-12)
# y_val_target   = np.log10(y_val + 1e-12)
# y_test_target  = np.log10(y_test + 1e-12)
# use_boxcox = False
# use_standardize = False

# ---------- 选项 1: Box-Cox 变换 ----------
y_train_bc, lambda_opt = boxcox(y_train + 1e-12)  # 自动寻找最佳 λ
y_val_bc   = (y_val + 1e-12) ** lambda_opt - 1 / lambda_opt
y_test_bc  = (y_test + 1e-12) ** lambda_opt - 1 / lambda_opt
y_train_target = y_train_bc
y_val_target   = y_val_bc
y_test_target  = y_test_bc
use_boxcox = True
use_standardize = False  # 下面可再叠加标准化

# ---------- 选项 2: 目标标准化（对 log10 或 Box-Cox 后的目标） ----------
# 若希望叠加标准化，将 use_standardize = True 并设置上面的 use_boxcox 为 True 或 False
# 示例：先 Box-Cox 再标准化
# scaler_y = StandardScaler()
# y_train_scaled = scaler_y.fit_transform(y_train_target.reshape(-1, 1)).ravel()
# y_val_scaled   = scaler_y.transform(y_val_target.reshape(-1, 1)).ravel()
# y_test_scaled  = scaler_y.transform(y_test_target.reshape(-1, 1)).ravel()
# y_train_target, y_val_target, y_test_target = y_train_scaled, y_val_scaled, y_test_scaled
# use_standardize = True

# ---------- 选项 3: 分位数回归（需配合目标变换） ----------
# 建议使用 log10 或 Box-Cox 后的目标，然后设置 objective='reg:quantileerror'
# ================================================================

# ========== 根据选择的变换，设置相应的逆变换函数 ==========
def inverse_transform(preds_target):
    # 如果使用了 Box-Cox
    if use_boxcox:
        preds_bc = preds_target
        # 如果同时使用了标准化，先逆标准化
        if use_standardize:
            preds_bc = scaler_y.inverse_transform(preds_bc.reshape(-1, 1)).ravel()
        preds = inv_boxcox(preds_bc, lambda_opt) - 1e-12
    else:
        # 仅 log10（可能加标准化）
        if use_standardize:
            preds_log = scaler_y.inverse_transform(preds_target.reshape(-1, 1)).ravel()
        else:
            preds_log = preds_target
        preds = 10 ** preds_log
    return preds

# ==================== 训练 XGBoost ====================
# 选择目标函数（可根据需要修改）
model = xgb.XGBRegressor(
    # objective='reg:absoluteerror',   # 原方案
    objective='reg:squarederror',      # 配合变换后常用
    # objective='reg:quantileerror',   # 分位数回归，需设置 quantile_alpha
    # quantile_alpha=0.5,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric='mae'
)

model.fit(
    X_train, y_train_target,
    eval_set=[(X_val, y_val_target)],
    verbose=False
)

# 预测
preds_val_target = model.predict(X_val)
preds_test_target = model.predict(X_test)

# 逆变换回原始延迟
preds_val = inverse_transform(preds_val_target)
preds_test = inverse_transform(preds_test_target)

# 计算相对误差
rel_err_val = np.abs(preds_val - y_val) / y_val * 100
rel_err_test = np.abs(preds_test - y_test) / y_test * 100

print(f"XGBoost Val Mean Relative Error: {np.mean(rel_err_val):.2f}%")
print(f"XGBoost Test Mean Relative Error: {np.mean(rel_err_test):.2f}%")
np.savez(os.path.join(OUTPUT_DIR, 'xgb_predictions.npz'), preds=preds_test, targets=y_test)