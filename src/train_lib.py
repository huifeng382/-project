import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hashlib
import torch
from src.lib_lookup import parse_lib, load_mapping, lookup_delay
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
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
    residuals = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            corner = data.corner_cond.to(device) if hasattr(data, 'corner_cond') else None
            csig = data.circuit_sig.to(device) if hasattr(data, 'circuit_sig') else None
            pred_log, _ = model(data.x, data.edge_index, data.batch, corner, csig)
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

def train_one_epoch(model, loader, optimizer, device, delta=1.0, show_progress=False,
                     lib_data=None):
    model.train()
    total_loss = 0
    total_batches = len(loader)
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        corner = data.corner_cond.to(device) if hasattr(data, 'corner_cond') else None
        csig = data.circuit_sig.to(device) if hasattr(data, 'circuit_sig') else None
        scalar_pred, node_sl = model(data.x, data.edge_index, data.batch, corner, csig)
        target_log = torch.log10(data.y + 1e-12)

        # 主 loss：GNN 标量预测
        residual = scalar_pred - target_log
        abs_res = torch.abs(residual)
        sample_loss = torch.where(abs_res <= delta,
                                  0.5 * residual ** 2,
                                  delta * (abs_res - 0.5 * delta))
        weights = torch.tensor([PIN_WEIGHTS.get(pin, 1.0) for pin in data.switching_pin], device=device)
        loss = (sample_loss * weights).mean()

        # 辅助 LIB loss：每节点 (slew, load) → 查表 → 和
        if lib_data is not None:
            gate_list, idx1_t, idx2_t, tables_t, mapping, gate_types_list = lib_data
            lib_loss = _compute_lib_loss(data, node_sl, gate_list, idx1_t, idx2_t,
                                          tables_t, mapping, gate_types_list, device)
            loss = loss + 0.01 * lib_loss  # LIB 辅助权重

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if show_progress and (i + 1) % 200 == 0:
            print(f"    batch {i+1}/{total_batches} ({100*(i+1)/total_batches:.0f}%)")
    return total_loss / len(loader)


def _compute_lib_loss(data, node_sl, gate_list, idx1_t, idx2_t, tables_t,
                       mapping, gate_types_list, device):
    """计算 LIB 查表延迟与真实延迟的 MSE"""
    from src.lib_lookup import lib_batch_lookup
    gate_mask = data.x[:, -1]
    total_lib_delay = torch.zeros(len(data.y), device=device)
    for gi in range(len(data.y)):
        g_mask = (data.batch == gi) & (gate_mask > 0.5)
        if g_mask.sum() == 0:
            continue
        gate_names = [gate_types_list[int(data.x[idx, 0].item())]
                      for idx in g_mask.nonzero(as_tuple=True)[0]]
        slew_pred = node_sl[g_mask, 0] * 50.0  # 缩放到 ps
        load_pred = node_sl[g_mask, 1] * 50.0  # 缩放到 fF
        d = lib_batch_lookup(gate_names, slew_pred, load_pred,
                              gate_list, idx1_t, idx2_t, tables_t, mapping)
        total_lib_delay[gi] = torch.sum(d) * 1e-12  # ps → s

    lib_log = torch.log10(total_lib_delay.clamp(1e-15))
    target_log = torch.log10(data.y + 1e-12)
    return torch.nn.functional.mse_loss(lib_log, target_log)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds_log = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            corner = data.corner_cond.to(device) if hasattr(data, 'corner_cond') else None
            csig = data.circuit_sig.to(device) if hasattr(data, 'circuit_sig') else None
            out, _ = model(data.x, data.edge_index, data.batch, corner, csig)
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

def _file_hash(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return 'none'

def _data_mtime_hash(static_parquets, dynamic_parquets):
    h = hashlib.md5()
    for p in sorted(static_parquets + dynamic_parquets):
        try:
            h.update(str(int(os.path.getmtime(p))).encode())
        except OSError:
            h.update(b'0')
    return h.hexdigest()[:8]

def _check_cache_dir(cache_subdir, version_key, description):
    """检查子缓存目录版本，过期则清除"""
    os.makedirs(cache_subdir, exist_ok=True)
    ver_file = os.path.join(cache_subdir, '.version')
    old_ver = None
    if os.path.exists(ver_file):
        with open(ver_file, 'r') as f:
            old_ver = f.read().strip()
    if old_ver != version_key:
        if old_ver:
            print(f"  {description}: outdated, clearing")
        else:
            print(f"  {description}: initializing")
        for f in os.listdir(cache_subdir):
            if f == '.version':
                continue
            fp = os.path.join(cache_subdir, f)
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
    with open(ver_file, 'w') as f:
        f.write(version_key)
    return old_ver == version_key  # True=命中, False=重建


def check_and_clear_cache(static_parquets=None, dynamic_parquets=None):
    """
    按缓存类型分别检查，仅在影响该类型的条件变化时清除。
    - 图缓存：graph_builder.py + 数据 mtime
    - 离群点缓存：model.py + graph_builder.py + 数据 mtime
    - gate 缓存：仅数据 mtime
    """
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gb_hash = _file_hash(os.path.join(src_dir, 'src/graph_builder.py'))
    model_hash = _file_hash(os.path.join(src_dir, 'src/model.py'))
    sim_hash = _file_hash(os.path.join(src_dir, 'src/logic_sim.py'))
    data_hash = _data_mtime_hash(static_parquets, dynamic_parquets)

    print("Cache check:")
    _check_cache_dir(os.path.join(CACHE_DIR, 'graphs'),
                      gb_hash + data_hash, "Graph cache")
    os.makedirs(os.path.join(CACHE_DIR, 'outlier'), exist_ok=True)
    # 离群点缓存自管理
    _check_cache_dir(os.path.join(CACHE_DIR, 'gate'),
                      sim_hash + data_hash, "Gate cache")
    sys.stdout.flush()


def get_outlier_cache_path(train_ids, static_parquets, dynamic_parquets):
    """生成离群点清洗缓存的路径，数据/模型/配置变化时自动失效。"""
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
    outlier_dir = os.path.join(CACHE_DIR, 'outlier')
    os.makedirs(outlier_dir, exist_ok=True)
    return os.path.join(outlier_dir, f'outlier_keep_{key_hash}.npy')


def main():
    t_total_start = time.time()
    set_seed(RANDOM_SEED)
    create_dir(CACHE_DIR)
    create_dir(OUTPUT_DIR)

    # ---------- 数据集路径：新生成 ~10w+ 样本 ----------
    # batch1: 手选电路全sweep (150电路, 30 corners) → ~58K
    # batch2: e-graph稀疏sweep (325电路, 9 corners) → ~43K
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_parquets = [
        os.path.join(data_dir, "data/batch1/circuit_static.parquet"),
        os.path.join(data_dir, "data/batch2_v4/circuit_static.parquet"),
    ]
    dynamic_parquets = [
        os.path.join(data_dir, "data/batch1/timing_arcs.parquet"),
        os.path.join(data_dir, "data/batch2_v4/timing_arcs.parquet"),
    ]
    # 可选追加数据（存在则加载，不存在则跳过）
    for batch in ['batch1b_v4', 'batch3_v4']:
        sp = os.path.join(data_dir, f"data/{batch}/circuit_static.parquet")
        dp = os.path.join(data_dir, f"data/{batch}/timing_arcs.parquet")
        if os.path.exists(sp) and os.path.exists(dp):
            static_parquets.append(sp)
            dynamic_parquets.append(dp)
            print(f"Found optional data: {batch}")
        else:
            print(f"Optional data not found, skipping: {batch}")
    # 启动时检查：如果代码或数据变了，自动清除过期缓存
    check_and_clear_cache(static_parquets, dynamic_parquets)

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

    # ---------- 可选：只保留4引脚标准电路（在划分前过滤，避免空split）----------
    if FOUR_PIN_ONLY:
        static_check = pd.concat([pd.read_parquet(p) for p in static_parquets])
        for col in ['candidate', 'candidate_id']:
            if col in static_check.columns:
                static_check = static_check.rename(columns={col: 'circuit_id'})
        static_check['circuit_id'] = static_check['circuit_id'].astype(str)
        four_pin_ids = set()
        for _, row in static_check.iterrows():
            try:
                pins = json.loads(row['input_pins_json']) if isinstance(row['input_pins_json'], str) else row['input_pins_json']
                if sorted(pins) == ['a', 'b', 'c', 'd']:
                    four_pin_ids.add(row['circuit_id'])
            except: pass
        old_n = len(dynamic_df)
        dynamic_df = dynamic_df[dynamic_df['circuit_id'].isin(four_pin_ids)]
        removed = 100 * (1 - len(four_pin_ids) / dynamic_df['circuit_id'].nunique())
        print(f"4-pin only: {dynamic_df['circuit_id'].nunique()} circuits, "
              f"samples={len(dynamic_df)} ({len(dynamic_df)/old_n*100:.0f}% of total)")

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
            # 匹配 data_loader 逻辑：优先 per-pin slew 列，否则只有切换引脚用全局 slew
            slew_col = f'slew_{pin}'
            if slew_col in row.index and pd.notna(row[slew_col]):
                slew_val = row[slew_col]
            elif pin == switching:
                slew_val = global_slew
            else:
                slew_val = 0.0
            # 匹配 data_loader 逻辑：优先 per-pin load，否则静态字典
            load_col = f'load_{pin}'
            if load_col in row.index and pd.notna(row[load_col]):
                load_val = row[load_col]
            else:
                load_val = loads_dict.get(pin, 0.0)
            # 匹配 data_loader 逻辑：仅切换引脚有 arrival_time
            if pin == switching:
                arrival_col = f'arrival_time_{pin}'
                if arrival_col in row.index and pd.notna(row[arrival_col]):
                    arrival_val = row[arrival_col]
                else:
                    arrival_val = row.get('arrival_time_s', 0.0)
            else:
                arrival_val = 0.0
            all_cont_features.append([slew_val, load_val, out_load, arrival_val])
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

    # 电路分组 Sampler：每批包含2-4个电路的所有corner，梯度混合多样本
    from torch.utils.data import Sampler
    class CircuitGroupSampler(Sampler):
        def __init__(self, dataset):
            cids = dataset.dynamic_df['circuit_id'].values
            self.circuit_groups = {}
            for i, c in enumerate(cids):
                self.circuit_groups.setdefault(c, []).append(i)
            self.circuits = list(self.circuit_groups.keys())
            self.n_samples = len(cids)
        def __iter__(self):
            np.random.shuffle(self.circuits)
            batches = []
            current = []
            for c in self.circuits:
                indices = self.circuit_groups[c]
                if len(current) + len(indices) > BATCH_SIZE * 2 and len(current) >= BATCH_SIZE:
                    batches.append(current)
                    current = []
                current.extend(indices)
            if current:
                batches.append(current)
            # 截断到 BATCH_SIZE（最后一批可能较大）
            result = []
            for b in batches:
                result.extend(b[:BATCH_SIZE * 2])  # 保留一些超额
            # 确保总样本数正确（取模以适配多个epoch）
            if len(result) < self.n_samples:
                result = result * (self.n_samples // len(result) + 1)
            return iter(result[:self.n_samples])
        def __len__(self):
            return self.n_samples
    sampler = CircuitGroupSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    sample_data = next(iter(train_loader))
    in_dim = sample_data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Input dim: {in_dim}, Device: {device}")

    # ---------- 离群点清洗 ----------
    if OUTLIER_CLEANING and len(train_dataset) > 100:
        # 测试模式下缩减离群点清洗，节省时间
        base_epochs = BASE_EPOCHS
        if QUICK_TEST:
            base_epochs = min(BASE_EPOCHS, 5)
            print(f"\nQUICK_TEST mode: reducing outlier cleaning to {base_epochs} epochs")

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
            base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            print(f"  Training base model on {len(train_dataset)} samples "
                  f"({len(base_loader)} batches/epoch, may take several minutes per epoch on CPU)...")
            best_base_loss = float('inf')
            base_patience_counter = 0
            for ep in range(base_epochs):
                loss = train_one_epoch(base_model, base_loader, base_optimizer, device,
                                       delta=HUBER_DELTA, show_progress=True)
                print(f"  Base epoch {ep+1}/{base_epochs}: loss = {loss:.4f}")
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
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    else:
        print("\n跳过离群点清洗（未启用或样本量过少）\n")

    # ---------- 加载 LIB 数据 ----------
    lib_data = None
    lib_path = os.path.join(data_dir, "data/std_cells.lib")
    map_path = os.path.join(data_dir, "data/sc_to_asap7.json")
    if os.path.exists(lib_path) and os.path.exists(map_path):
        lib = parse_lib(lib_path)
        mapping = load_mapping(map_path)
        gate_list, idx1_t, idx2_t, tables_t = build_lib_tensors(lib, mapping)
        gate_types_list = [gb.GATE_TYPES[i] if i < len(gb.GATE_TYPES) else 'OTHER'
                           for i in range(num_gate_types)]
        lib_data = (gate_list, idx1_t, idx2_t, tables_t, mapping, gate_types_list)
        print(f"LIB loaded: {len(gate_list)} cell types, {sum(1 for v in mapping.values() if v)} SC mapped")
    else:
        print("LIB or mapping not found, using GNN-only mode")

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
    plateau_counter = 0
    val_err_history = []
    train_loss_history = []
    plateau_triggered = False
    last_lr = LEARNING_RATE
    lr_decayed = False
    print("\nStart training...")
    t_train_start = time.time()
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, delta=HUBER_DELTA,
                                      lib_data=lib_data)
        val_loss, val_rel_err, _, _ = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Rel Err: {val_rel_err:.2f}%")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        val_err_history.append(val_rel_err)
        train_loss_history.append(train_loss)

        # 检测 LR 是否已衰减（LR 降低是突破平台期的契机，在此之前不早停）
        if current_lr < last_lr * 0.99:
            lr_decayed = True
        last_lr = current_lr

        if val_rel_err < best_val_rel:
            best_val_rel = val_rel_err
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            patience_counter = 0
            plateau_counter = 0
            print(f"  >>> New best model saved (Val Rel Err: {val_rel_err:.2f}%)")
        else:
            patience_counter += 1
            plateau_counter += 1

            # ---- 测试模式：快速平台检测 ----
            if QUICK_TEST and epoch + 1 >= QUICK_MIN_EPOCHS:
                if len(val_err_history) > QUICK_WINDOW:
                    best_before_window = min(val_err_history[:-QUICK_WINDOW])
                    improved = best_before_window - best_val_rel
                    if improved < QUICK_MIN_DELTA:
                        print(f"  >>> Quick test stop: best={best_val_rel:.1f}%, "
                              f"only improved {improved:.1f} pts in last {QUICK_WINDOW} epochs")
                        break

            # ---- 智能早停：检测过拟合平台期 ----
            # 仅当 LR 已衰减过 + train 还在降 + val 不再改善 → 过拟合，提前终止
            if (plateau_counter >= PLATEAU_WINDOW
                    and epoch + 1 >= PLATEAU_MIN_EPOCHS
                    and lr_decayed
                    and not plateau_triggered):
                recent_val = val_err_history[-PLATEAU_WINDOW:]
                recent_train = train_loss_history[-PLATEAU_WINDOW:]
                # 与更早的窗口比较
                prev_start = max(0, len(train_loss_history) - 2 * PLATEAU_WINDOW)
                prev_train = train_loss_history[prev_start:len(train_loss_history) - PLATEAU_WINDOW]

                val_range = max(recent_val) - min(recent_val)
                val_best_recent = min(recent_val)
                train_mean_recent = np.mean(recent_train)
                train_mean_prev = np.mean(prev_train) if prev_train else train_mean_recent

                # 条件1: train loss 在持续下降（比前一窗口明显更低）
                train_still_improving = train_mean_recent < train_mean_prev - 0.0005
                # 条件2: val err 没有破新低（比全局最优差超过 PLATEAU_MIN_DELTA 个百分点）
                val_not_improving = val_best_recent > best_val_rel + PLATEAU_MIN_DELTA
                # 条件3: val err 在窗口内震荡（没有明显下降趋势）
                val_first_half = np.mean(recent_val[:PLATEAU_WINDOW//2])
                val_second_half = np.mean(recent_val[PLATEAU_WINDOW//2:])
                val_no_trend = val_second_half > val_first_half - PLATEAU_MIN_DELTA

                if train_still_improving and val_not_improving and val_no_trend:
                    plateau_triggered = True
                    print(f"  >>> Plateau detected: train still improving ({train_mean_prev:.4f}→{train_mean_recent:.4f}) "
                          f"but val oscillating in [{min(recent_val):.1f}%~{max(recent_val):.1f}%], best={best_val_rel:.1f}%")
                    break

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
        def _expr_num(e):
            try:
                return int(str(e).replace('expr', ''))
            except:
                return -1
        expr_nums = test_dyn['expr'].apply(_expr_num).values
        b1_mask = (expr_nums >= 0) & (expr_nums <= 199)     # batch1 + batch1b
        b2_mask = (expr_nums >= 200) & (expr_nums <= 999)   # batch2
        b3_mask = expr_nums >= 1000                           # batch3
        for label, mask in [('B1(全sweep)', b1_mask), ('B2(稀疏)', b2_mask), ('B3(新建)', b3_mask)]:
            if mask.sum() > 0:
                err = np.abs(preds[mask] - targets[mask]) / targets[mask] * 100
                print(f"  {label}: n={mask.sum():,}  mean_err={np.mean(err):.1f}%")

    # ---------- 摘要 ----------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    import subprocess
    try:
        ver = subprocess.check_output(
            ['git', 'log', '--oneline', '-1'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True, stderr=subprocess.DEVNULL).strip()
    except:
        ver = 'unknown'
    print(f"  Version: {ver}")
    print(f"  Config: HIDDEN_DIM={HIDDEN_DIM} NUM_LAYERS={NUM_LAYERS} "
          f"DROPOUT={DROPOUT} WEIGHT_DECAY={WEIGHT_DECAY}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Best Val Rel Err: {best_val_rel:.2f}%")
    print(f"  Test Rel Err: {test_rel_err:.2f}%")
    if 'corner' in test_dyn.columns:
        corners = test_dyn['corner'].values
        corner_errs = {}
        for c in sorted(set(corners)):
            mask = corners == c
            if mask.sum() > 0:
                corner_errs[c] = np.mean(np.abs(preds[mask] - targets[mask]) / targets[mask] * 100)
        best_c = min(corner_errs, key=corner_errs.get)
        worst_c = max(corner_errs, key=corner_errs.get)
        print(f"  Best corner: {best_c} = {corner_errs[best_c]:.1f}%")
        print(f"  Worst corner: {worst_c} = {corner_errs[worst_c]:.1f}%")
        print(f"  Corner spread: {corner_errs[worst_c] - corner_errs[best_c]:.1f}%")
    # 批次误差（安全的，处理 expr 不存在的情况）
    if 'expr' in test_dyn.columns:
        for label, mask in [('B1(全sweep)', b1_mask), ('B2(稀疏)', b2_mask), ('B3(新建)', b3_mask)]:
            if mask.sum() > 0:
                err = np.mean(np.abs(preds[mask] - targets[mask]) / targets[mask] * 100)
                print(f"  {label}: {err:.1f}%")
    print(f"  Total samples: test={len(test_dyn)} train={len(train_dataset)} val={len(val_dataset)}")
    t_total = time.time() - t_total_start
    t_train = time.time() - t_train_start
    avg_epoch = t_train / (epoch + 1)
    print(f"  Total time: {t_total/60:.1f} min | Train time: {t_train/60:.1f} min "
          f"| Avg/epoch: {avg_epoch:.1f}s ({epoch+1} epochs)")
    print("=" * 60)

if __name__ == "__main__":
    main()
