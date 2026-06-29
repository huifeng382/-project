import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hashlib
import shutil
import torch.utils.data
from config import *
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from config import HUBER_DELTA
from src.utils import set_seed, split_by_circuit, save_scaler, create_dir
from src.data_loader import DelayDataset
from src.model import DelayGNN
from src.graph_builder import rebuild_gate_types

PIN_WEIGHTS = {'a': 1.3, 'b': 1.0, 'c': 1.0, 'd': 1.3, 'e': 1.0}

def log_mse_loss(pred_log, target):
    target_log = torch.log10(target + 1e-12)
    return F.mse_loss(pred_log, target_log)

def get_train_residuals(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
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
    residuals = get_train_residuals(model, dataset, device)
    threshold = np.percentile(residuals, 100 - top_percent)
    keep_indices = np.where(residuals <= threshold)[0].tolist()
    print(f"清洗前样本数: {len(dataset)}, 清洗后: {len(keep_indices)}, 剔除比例: {100 - len(keep_indices)/len(dataset)*100:.1f}%")
    return torch.utils.data.Subset(dataset, keep_indices)

def train_one_epoch(model, loader, optimizer, device, delta=1.0):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        target_log = torch.log10(data.y + 1e-12)
        residual = out - target_log
        abs_res = torch.abs(residual)
        sample_loss = torch.where(abs_res <= delta,
                                  0.5 * residual ** 2,
                                  delta * (abs_res - 0.5 * delta))
        weights = torch.tensor([PIN_WEIGHTS.get(pin, 1.0) for pin in data.switching_pin], device=device)
        loss = (sample_loss * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds_log = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = log_mse_loss(out, data.y)
            total_loss += loss.item()
            preds_log.append(out.cpu().numpy())
            targets.append(data.y.cpu().numpy())
    if len(preds_log) == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    preds_log = np.concatenate(preds_log)
    targets = np.concatenate(targets)
    preds = 10 ** preds_log
    preds = np.clip(preds, 1e-12, 1e-8)
    rel_error = np.abs(preds - targets) / targets * 100
    return total_loss / len(loader), np.mean(rel_error), preds, targets

def get_outlier_cache_path(train_ids, static_parquets, dynamic_parquets):
    """生成离群点清洗缓存的路径，数据或配置变化时自动失效。"""
    key_parts = [
        ','.join(sorted(train_ids)),
        f"top{OUTLIER_TOP_PERCENT}",
        f"base{BASE_EPOCHS}",
        f"huber{HUBER_DELTA}",
        f"seed{RANDOM_SEED}",
        f"hdim{HIDDEN_DIM}",
        f"nlay{NUM_LAYERS}",
    ]
    # 加入数据文件的修改时间，数据变了缓存自动失效
    for p in sorted(static_parquets + dynamic_parquets):
        try:
            key_parts.append(str(int(os.path.getmtime(p))))
        except OSError:
            key_parts.append('0')
    key_str = '_'.join(key_parts)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f'outlier_keep_{key_hash}.npy')


def main():
    set_seed(RANDOM_SEED)
    create_dir(CACHE_DIR)
    create_dir(OUTPUT_DIR)

    # ---------- 数据集路径：方案B采样后 ~10万样本 ----------
    # batch1: 手选电路全sweep (170电路, 30 corners) → ~30K
    # batch2: e-graph稀疏sweep (1215电路, 9 corners) → ~70K
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_parquets = [
        os.path.join(data_dir, "data/batch1_30k/circuit_static.parquet"),
        os.path.join(data_dir, "data/batch2_70k/circuit_static.parquet"),
    ]
    dynamic_parquets = [
        os.path.join(data_dir, "data/batch1_30k/timing_arcs.parquet"),
        os.path.join(data_dir, "data/batch2_70k/timing_arcs.parquet"),
    ]

    for p in static_parquets + dynamic_parquets:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    dynamic_dfs = [pd.read_parquet(p) for p in dynamic_parquets]
    dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)

    # ---------- 列名规范化：合并 candidate_id → circuit_id ----------
    if 'candidate_id' in dynamic_df.columns:
        if 'circuit_id' not in dynamic_df.columns:
            dynamic_df = dynamic_df.rename(columns={'candidate_id': 'circuit_id'})
        else:
            dynamic_df['circuit_id'] = dynamic_df['circuit_id'].fillna(
                dynamic_df['candidate_id'].astype(str))
            dynamic_df = dynamic_df.drop(columns=['candidate_id'])
    # 合并 delay_s → DELAY
    if 'delay_s' in dynamic_df.columns:
        if 'DELAY' not in dynamic_df.columns:
            dynamic_df = dynamic_df.rename(columns={'delay_s': 'DELAY'})
        else:
            dynamic_df['DELAY'] = dynamic_df['DELAY'].fillna(dynamic_df['delay_s'])
            dynamic_df = dynamic_df.drop(columns=['delay_s'])

    # ---------- 数据清洗 ----------
    print(f"原始样本数: {len(dynamic_df)}")
    dynamic_df = dynamic_df.dropna(subset=['circuit_id', 'DELAY'])
    dynamic_df['circuit_id'] = dynamic_df['circuit_id'].astype(str)
    dynamic_df = dynamic_df[(dynamic_df['DELAY'] > 1e-12) & (dynamic_df['DELAY'] < 1e-8)]
    print(f"清洗后样本数: {len(dynamic_df)}, 电路数: {dynamic_df['circuit_id'].nunique()}")

    circuit_ids = dynamic_df['circuit_id'].unique().tolist()
    train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)
    print(f"划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} 电路")

    # ---------- 读取静态数据用于 scaler ----------
    static_dfs_raw = [pd.read_parquet(p) for p in static_parquets]
    for i, df in enumerate(static_dfs_raw):
        # 列名规范化
        for col in ['candidate', 'candidate_id']:
            if col in df.columns:
                df = df.rename(columns={col: 'circuit_id'})
        df['circuit_id'] = df['circuit_id'].astype(str)
        # 优先使用标准化网表
        if 'gate_level_netlist_std' in df.columns:
            df = df.drop(columns=['gate_level_netlist'], errors='ignore')
            df = df.rename(columns={'gate_level_netlist_std': 'gate_level_netlist'})
        static_dfs_raw[i] = df
    static_df = pd.concat(static_dfs_raw).drop_duplicates('circuit_id').set_index('circuit_id')
    pin_loads_map = {}
    for cid, srow in static_df.iterrows():
        try:
            pin_loads_map[cid] = json.loads(srow['pin_loads_json'])
        except Exception:
            pin_loads_map[cid] = {}

    # 根据 dynamic_df 列推断引脚（排除 slew_s，它是全局值不是引脚）
    pins = sorted([c[5:] for c in dynamic_df.columns if c.startswith('slew_') and c != 'slew_s'])
    actual = set(dynamic_df['switching_pin'].dropna().unique())
    pins = [p for p in pins if p in actual]
    if not pins:
        pins = sorted(actual)
    print(f"引脚: {pins}")

    # ---------- Scaler 拟合（匹配 DelayDataset._get_dynamic_features 逻辑）----------
    train_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)]
    all_cont_features = []
    for _, row in train_dynamic.iterrows():
        switching = row.get('switching_pin', '')
        global_slew = row.get('slew_s', 0.0)
        out_load = row.get('output_load_f', 0.0)
        loads_dict = pin_loads_map.get(row['circuit_id'], {})
        for pin in pins:
            # 匹配 data_loader 逻辑：优先 per-pin slew，否则所有节点共享全局 slew
            slew_col = f'slew_{pin}'
            if slew_col in row.index and pd.notna(row[slew_col]):
                slew_val = row[slew_col]
            else:
                slew_val = global_slew
            # 匹配 data_loader 逻辑：优先 per-pin load，否则静态字典
            load_col = f'load_{pin}'
            if load_col in row.index and pd.notna(row[load_col]):
                load_val = row[load_col]
            else:
                load_val = loads_dict.get(pin, 0.0)
            all_cont_features.append([slew_val, load_val, out_load])
    scaler = StandardScaler(with_std=True)
    scaler.fit(all_cont_features)
    print("=" * 50)
    print("Scaler check:")
    print(f"  Mean: {scaler.mean_}")
    print(f"  Scale (std): {scaler.scale_}")
    if (scaler.scale_ == 0).any():
        print("  WARNING: Some features have zero variance!")
    print("=" * 50)
    save_scaler(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))

    # ---------- 动态构建门类型映射 + 清除旧缓存 ----------
    # 收集所有数据中的 cell 类型，更新 graph_builder 的 GATE_TYPES
    all_cell_types = set()
    for _, srow in static_df.iterrows():
        try:
            types = json.loads(srow['cell_types_json']) if isinstance(srow['cell_types_json'], str) else srow['cell_types_json']
            all_cell_types.update(types)
        except Exception:
            pass
    rebuild_gate_types(all_cell_types)
    import src.graph_builder as gb
    num_gate_types = len(gb.GATE_TYPES)
    print(f"Gate types: {len(all_cell_types)} unique cell types -> gate vocabulary rebuilt ({num_gate_types} total)")

    # 清除旧图缓存（GATE_TYPES 变了，one-hot 维度不同）
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---------- 创建数据集 ----------
    train_dataset = DelayDataset(static_parquets, dynamic_parquets, train_ids, scaler, CACHE_DIR)
    val_dataset = DelayDataset(static_parquets, dynamic_parquets, val_ids, scaler, CACHE_DIR)
    test_dataset = DelayDataset(static_parquets, dynamic_parquets, test_ids, scaler, CACHE_DIR)
    print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # 空 edge_index 检查
    print("Checking test dataset for empty edge_index...")
    empty_circuits = set()
    for idx in range(len(test_dataset)):
        data = test_dataset[idx]
        if data.edge_index.numel() == 0:
            cid = test_dataset.dynamic_df.iloc[idx]['circuit_id']
            empty_circuits.add(cid)
    if empty_circuits:
        print(f"WARNING: {len(empty_circuits)} circuits with empty edge_index: {empty_circuits}")
    else:
        print("No empty edge_index found.")

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

    sample_data = next(iter(train_loader))
    in_dim = sample_data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Input dim: {in_dim}, Device: {device}")

    # ---------- 离群点清洗 ----------
    if OUTLIER_CLEANING and len(train_dataset) > 100:
        cache_path = get_outlier_cache_path(train_ids, static_parquets, dynamic_parquets)

        if os.path.exists(cache_path):
            keep_indices = np.load(cache_path).tolist()
            print(f"\n加载离群点清洗缓存: {cache_path}")
            print(f"  原始样本数: {len(train_dataset)}, 清洗后: {len(keep_indices)}, "
                  f"剔除: {(1 - len(keep_indices)/len(train_dataset))*100:.1f}%")
        else:
            print("\n========== 开始离群点清洗 ==========")
            base_model = DelayGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM,
                                  num_layers=NUM_LAYERS, dropout=DROPOUT,
                                  num_gate_types=num_gate_types,
                                  gate_embed_dim=GATE_EMBED_DIM).to(device)
            base_optimizer = Adam(base_model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
            base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            best_base_loss = float('inf')
            base_patience_counter = 0
            for ep in range(BASE_EPOCHS):
                loss = train_one_epoch(base_model, base_loader, base_optimizer, device, delta=HUBER_DELTA)
                print(f"  Base epoch {ep+1}/{BASE_EPOCHS}: loss = {loss:.4f}")
                # 动态早停
                if loss < best_base_loss - BASE_MIN_DELTA:
                    best_base_loss = loss
                    base_patience_counter = 0
                else:
                    base_patience_counter += 1
                if ep + 1 >= BASE_MIN_EPOCHS and base_patience_counter >= BASE_PATIENCE:
                    print(f"  离群点清洗早停于 epoch {ep+1}（loss 已连续 {BASE_PATIENCE} epoch 无明显下降）")
                    break

            residuals = get_train_residuals(base_model, train_dataset, device)
            threshold = np.percentile(residuals, 100 - OUTLIER_TOP_PERCENT)
            keep_indices = np.where(residuals <= threshold)[0].tolist()

            print(f"  原始样本数: {len(train_dataset)}")
            print(f"  清洗后样本数: {len(keep_indices)}")
            print(f"  剔除比例: {(1 - len(keep_indices)/len(train_dataset))*100:.1f}%")

            np.save(cache_path, np.array(keep_indices))
            print(f"  缓存已保存: {cache_path}")

            del base_model, base_optimizer
            print("========== 清洗完成 ==========\n")

        train_subset = torch.utils.data.Subset(train_dataset, keep_indices)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        print("\n跳过离群点清洗（未启用或样本量过少）\n")

    model = DelayGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
                     num_gate_types=num_gate_types,
                     gate_embed_dim=GATE_EMBED_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if LR_SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=LR_MIN, cooldown=LR_COOLDOWN
        )
    elif LR_SCHEDULER == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=LR_T_MAX, eta_min=LR_ETA_MIN
        )

    best_val_rel = float('inf')
    patience_counter = 0
    print("\nStart training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, delta=HUBER_DELTA)
        val_loss, val_rel_err, _, _ = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Rel Err: {val_rel_err:.2f}%")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_rel_err < best_val_rel:
            best_val_rel = val_rel_err
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            patience_counter = 0
            print(f"  >>> New best model saved (Val Rel Err: {val_rel_err:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
    val_loss, val_rel_err, _, _ = evaluate(model, val_loader, device)
    print(f"Best model on Val: Loss = {val_loss:.4f} | Rel Err = {val_rel_err:.2f}%")
    test_loss, test_rel_err, preds, targets = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Mean Relative Error: {test_rel_err:.2f}%")
    np.savez(os.path.join(OUTPUT_DIR, 'test_predictions.npz'), preds=preds, targets=targets)

    # Per-corner breakdown (使用 test_dataset 的 dynamic_df 保证行数对齐)
    test_dyn = test_dataset.dynamic_df.reset_index(drop=True)
    if 'corner' in test_dyn.columns and len(preds) > 0:
        print(f"\nPer-corner relative error (test samples: {len(test_dyn)}, preds: {len(preds)}):")
        if len(test_dyn) == len(preds):
            corners = test_dyn['corner'].values
            for c in sorted(set(corners)):
                mask = corners == c
                if mask.sum() > 0:
                    err = np.abs(preds[mask] - targets[mask]) / targets[mask] * 100
                    print(f"  {c}: n={mask.sum():,}  mean_err={np.mean(err):.1f}%")
        else:
            print(f"  WARNING: row mismatch (test_dyn={len(test_dyn)}, preds={len(preds)})")

    # Per-batch breakdown
    if 'expr' in test_dyn.columns:
        # batch1 circuits have expr starting with expr04, batch2 start with expr00
        batch1_mask = test_dyn['expr'].str.startswith('expr04').values
        batch2_mask = ~batch1_mask
        for label, mask in [('Batch1 (handpicked)', batch1_mask), ('Batch2 (egraph)', batch2_mask)]:
            if mask.sum() > 0:
                err = np.abs(preds[mask] - targets[mask]) / targets[mask] * 100
                print(f"  {label}: n={mask.sum():,}  mean_err={np.mean(err):.1f}%")

if __name__ == "__main__":
    main()
