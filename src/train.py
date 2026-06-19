import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.utils.data
from config import *
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.utils import set_seed, split_by_circuit, save_scaler, create_dir
from src.data_loader import DelayDataset
from src.model import DelayGNN
PIN_WEIGHTS = {
    'a': 1.3,
    'b': 1.0,
    'c': 1.0,
    'd': 1.3,
    'e': 1.0,
}
def log_mse_loss(pred_log, target):
    """pred_log: 模型输出的 log10(delay) , target: 真实 delay"""
    target_log = torch.log10(target + 1e-12)
    return F.mse_loss(pred_log, target_log)

def get_train_residuals(model, dataset, device):
    """用当前模型计算训练集每个样本的残差（log10域）"""
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)  # 大batch加速
    residuals = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred_log = model(data.x, data.edge_index, data.batch)
            target_log = torch.log10(data.y + 1e-12)
            res = torch.abs(pred_log - target_log).cpu().numpy()
            residuals.extend(res)
    return np.array(residuals)

def clean_outliers_by_residual(dataset, model, device, top_percent=5):
    """
    剔除训练集中残差最大的 top_percent% 样本，返回 Subset
    """
    residuals = get_train_residuals(model, dataset, device)
    # 计算阈值（百分位数）
    threshold = np.percentile(residuals, 100 - top_percent)
    keep_indices = np.where(residuals <= threshold)[0].tolist()
    print(f"清洗前样本数: {len(dataset)}, 清洗后: {len(keep_indices)}, 剔除比例: {100 - len(keep_indices)/len(dataset)*100:.1f}%")
    return torch.utils.data.Subset(dataset, keep_indices)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # 基础 loss
        target_log = torch.log10(data.y + 1e-12)
        base_loss = F.mse_loss(out, target_log)
        # 获取每个样本的权重
        weights = torch.tensor([PIN_WEIGHTS[pin] for pin in data.switching_pin], device=device)
        # 加权 loss：对每个样本的 MSE 加权平均
        # 注意：out 和 target_log 都是 batch 中的每个样本
        # 需要逐元素计算平方差后加权平均
        squared_diff = (out - target_log) ** 2
        weighted_loss = (squared_diff * weights).mean()
        # 或使用加权 MSE
        loss = weighted_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds_log, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)   # out 是 log10(delay)
            loss = log_mse_loss(out, data.y)
            total_loss += loss.item()
            preds_log.append(out.cpu().numpy())
            targets.append(data.y.cpu().numpy())
    preds_log = np.concatenate(preds_log)
    targets = np.concatenate(targets)
    preds = 10 ** preds_log           # 转换回实际延迟
    rel_error = np.abs(preds - targets) / targets * 100
    return total_loss / len(loader), np.mean(rel_error), preds, targets

def main():

    set_seed(RANDOM_SEED)
    create_dir(CACHE_DIR)
    create_dir(OUTPUT_DIR)

    # 自动扫描所有批次的 Parquet 文件
    static_parquets = glob.glob("data/batch_*/circuit_static.parquet")
    dynamic_parquets = glob.glob("data/batch_*/timing_arcs.parquet")

    if not static_parquets or not dynamic_parquets:
        raise FileNotFoundError(
            "No Parquet files found in data/batch_*/ directories.\n"
            "Please run convert_to_parquet.py first to generate the files."
        )

    # 读取所有动态数据并合并（用于获取 circuit_ids 和标准化）
    dynamic_dfs = [pd.read_parquet(p) for p in dynamic_parquets]
    dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)

    # ========== 数据清洗开始 ==========
    original_len = len(dynamic_df)
    dynamic_df = dynamic_df.dropna(subset=['circuit_id'])
    dynamic_df['circuit_id'] = dynamic_df['circuit_id'].astype(str)
    dynamic_df = dynamic_df.dropna(subset=['DELAY'])
    removed = original_len - len(dynamic_df)
    if removed > 0:
        print(f"Data cleaning: removed {removed} rows with NaN values.")
    print(f"Remaining rows: {len(dynamic_df)}")
    # ========== 数据清洗结束 ==========
    print("Unique circuit_ids after cleaning:", dynamic_df['circuit_id'].nunique())

    circuit_ids = dynamic_df['circuit_id'].unique().tolist()

    # 按电路划分训练/验证/测试集
    train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)

    # 准备标准化器（基于训练集中所有电路的各引脚 slew, arrival, load）
    train_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)]
    all_cont_features = []
    pins = ['a','b','c','d','e']
    for _, row in train_dynamic.iterrows():
        for pin in pins:
            all_cont_features.append([row[f'slew_{pin}'], row[f'arrival_{pin}'], row[f'load_{pin}']])
    scaler = StandardScaler(with_std=False)
    scaler.fit(all_cont_features)
    print("=" * 50)
    print("Scaler check:")
    print(f"  Mean shape: {scaler.mean_.shape if scaler.mean_ is not None else 'None'}")
    if scaler.scale_ is not None:
        print(f"  Scale (std) shape: {scaler.scale_.shape}")
        print(f"  Scale values: {scaler.scale_}")
        if (scaler.scale_ == 0).any():
            print("  WARNING: Some features have zero variance!")
    else:
        print("  Scale (std): None (using with_std=False)")
    print("=" * 50)
    save_scaler(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    # 创建数据集（传入文件列表）
    train_dataset = DelayDataset(
        static_parquets=static_parquets,
        dynamic_parquets=dynamic_parquets,
        circuit_ids=train_ids,
        scaler=scaler,
        cache_dir=CACHE_DIR
    )

    val_dataset = DelayDataset(
        static_parquets=static_parquets,
        dynamic_parquets=dynamic_parquets,
        circuit_ids=val_ids,
        scaler=scaler,
        cache_dir=CACHE_DIR
    )
    test_dataset = DelayDataset(
        static_parquets=static_parquets,
        dynamic_parquets=dynamic_parquets,
        circuit_ids=test_ids,
        scaler=scaler,
        cache_dir=CACHE_DIR
    )
    print("Checking test dataset for empty edge_index...")
    empty_circuits = []
    for idx in range(len(test_dataset)):
        data = test_dataset[idx]
        if data.edge_index.numel() == 0:
            row = test_dataset.dynamic_df.iloc[idx]
            cid = row['circuit_id']
            empty_circuits.append(cid)
            print(f"WARNING: circuit {cid} (sample {idx}) has empty edge_index")

    if empty_circuits:
        print(f"Found {len(set(empty_circuits))} unique circuits with empty edge_index: {set(empty_circuits)}")
    else:
        print("No empty edge_index found.")
    # 在 train_dataset 创建后立即添加
    sample = train_dataset[0]
    print("=" * 50)
    print("Data check:")
    print(f"  Sample x shape: {sample.x.shape}")
    print(f"  Sample x min: {sample.x.min().item():.3e}")
    print(f"  Sample x max: {sample.x.max().item():.3e}")
    print(f"  Sample x has nan: {torch.isnan(sample.x).any().item()}")
    print(f"  Sample x has inf: {torch.isinf(sample.x).any().item()}")
    print(f"  Sample y: {sample.y.item():.3e}")
    print("=" * 50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 获取输入特征维度
    sample_data = next(iter(train_loader))
    in_dim = sample_data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================== 新增：离群点清洗 ====================
    # 可配置参数（您可调整或移至 config.py）
    OUTLIER_CLEANING = True      # 是否启用清洗
    OUTLIER_TOP_PERCENT = 5      # 剔除残差最大的前百分之几
    BASE_EPOCHS = 20             # 基准模型训练轮数（无需收敛）

    if OUTLIER_CLEANING and len(train_dataset) > 100:   # 数据量过小则跳过
        print("\n========== 开始离群点清洗 ==========")
        # 1. 训练基准模型（快速）
        base_model = DelayGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM,
                              num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
        base_optimizer = Adam(base_model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
        base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        for ep in range(BASE_EPOCHS):
            loss = train_one_epoch(base_model, base_loader, base_optimizer, device)
            print(f"  Base epoch {ep+1}/{BASE_EPOCHS}: loss = {loss:.4f}")

        # 2. 计算每个样本的残差
        residuals = get_train_residuals(base_model, train_dataset, device)
        threshold = np.percentile(residuals, 100 - OUTLIER_TOP_PERCENT)
        keep_indices = np.where(residuals <= threshold)[0].tolist()

        print(f"  原始样本数: {len(train_dataset)}")
        print(f"  清洗后样本数: {len(keep_indices)}")
        print(f"  剔除比例: {(1 - len(keep_indices)/len(train_dataset))*100:.1f}%")

        # 3. 用清洗后的子集替换原训练集，重新构建 train_loader
        train_subset = torch.utils.data.Subset(train_dataset, keep_indices)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        # 释放基准模型显存（如有）
        del base_model, base_optimizer
        print("========== 清洗完成 ==========\n")
    else:
        print("\n跳过离群点清洗（未启用或样本量过少）\n")
    # ==================== 清洗结束 ====================    
    
    model = DelayGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---------- 新增调度器 ----------
    scheduler = None
    if LR_SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=LR_MIN, cooldown=LR_COOLDOWN
        )
    elif LR_SCHEDULER == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # --------------------------------

    # ========== 修改点：保存和早停依据改为 Val Rel Err ==========
    best_val_rel = float('inf')           # <<< 改为追踪相对误差
    patience_counter = 0
    print("Start training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_rel_err, _, _ = evaluate(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Rel Err: {val_rel_err:.2f}%")
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 保存和早停条件基于 Val Rel Err
        if val_rel_err < best_val_rel:    # <<< 保存条件改为相对误差
            best_val_rel = val_rel_err
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            patience_counter = 0
            print(f"  >>> New best model saved (Val Rel Err: {val_rel_err:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break
    # ======================================================

    # 测试最佳模型
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
    val_loss_after_load, val_rel_err_after_load, _, _ = evaluate(model, val_loader, device)
    print(f"Best model performance on Val Set: Loss = {val_loss_after_load:.4f} | Rel Err = {val_rel_err_after_load:.2f}%")
    test_loss, test_rel_err, preds, targets = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Mean Relative Error: {test_rel_err:.2f}%")
    np.savez(os.path.join(OUTPUT_DIR, 'test_predictions.npz'), preds=preds, targets=targets)
if __name__ == "__main__":
    main()