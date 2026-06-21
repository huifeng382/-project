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

# ---- 离群点清洗（方案二） ----
lower = np.percentile(y_train, 1)
upper = np.percentile(y_train, 99)
keep = (y_train > lower) & (y_train < upper)
X_train = X_train[keep]
y_train = y_train[keep]
print(f"离群点清洗后训练集大小: {len(X_train)}")
# -----------------------------

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

y_train_log = np.log10(y_train + 1e-12)
y_val_log = np.log10(y_val + 1e-12)
y_test_log = np.log10(y_test + 1e-12)

# ---- 使用鲁棒目标函数（方案三） ----
model = xgb.XGBRegressor(
    objective='reg:absoluteerror',  # 对离群点更鲁棒
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
    X_train, y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False
)

preds_val_log = model.predict(X_val)
preds_test_log = model.predict(X_test)
preds_val = 10 ** preds_val_log
preds_test = 10 ** preds_test_log

rel_err_val = np.abs(preds_val - y_val) / y_val * 100
rel_err_test = np.abs(preds_test - y_test) / y_test * 100

print(f"XGBoost Val Mean Relative Error: {np.mean(rel_err_val):.2f}%")
print(f"XGBoost Test Mean Relative Error: {np.mean(rel_err_test):.2f}%")
np.savez(os.path.join(OUTPUT_DIR, 'xgb_predictions.npz'), preds=preds_test, targets=y_test)