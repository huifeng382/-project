# utils.py
import os
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_by_circuit(circuit_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按电路划分训练/验证/测试集，保持电路完整性"""
    set_seed(seed)
    ids = list(circuit_ids)
    random.shuffle(ids)
    n = len(ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    return train_ids, val_ids, test_ids

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def create_dir(path):
    os.makedirs(path, exist_ok=True)