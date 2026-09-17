import sys
import os
import time
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
from src.utils import set_seed, split_by_circuit, split_by_expr, ranking_metrics, save_scaler, create_dir
from src.data_loader import DelayDataset
from src.model import DelayGNN
from src.graph_builder import rebuild_gate_types

PIN_WEIGHTS = {'a': 1.3, 'b': 1.0, 'c': 1.0, 'd': 1.3, 'e': 1.0}


# ---------- 部署选点政策（OPERATIONS §6.7:212 落地，17.4.1）----------
# 政策原文：「取平台末端（最后一个 midpoint），不取 shadow argmax；best_model.pt 不进部署候选」。
# 两个调用点（训练内 midpoint 块 / EVAL_ONLY=midpoint）**共用这两个函数**——历史上它们是两份
#   各自 argmax 的复制品，只改一处等于政策落一半（2026-09-17 就是这么发现 EVAL_ONLY 那条的）。
def midpoint_epoch_of(path):
    """midpoint_ep{N}.pt → N（int）；解析不出返回 -1。
    ⚠ 必要性：`sorted(glob(...))` 是**字符串序**，midpoint_ep100 会排在 midpoint_ep50 前面
    → 想「取末端」却直接取 `mid_files[-1]` 会取到 ep50（取反）。故排序与解析都必须走数值。"""
    try:
        return int(os.path.basename(path).replace('midpoint_ep', '').replace('.pt', ''))
    except Exception:
        return -1


def pick_midpoint(pairs, mode='last'):
    """从 [(ep, score), ...] 里按政策选一个 epoch。pairs 须**已按 ep 升序**。

    返回 (选中ep | None, 说明行列表)。**只做决策、不打印主日志**：两处调用点的逐 epoch
    打印格式不同（训练内是 `  ep  N: cap2=...`，eval-only 是 `[eval-only] epN(val): cap2=...`），
    各自保留原格式不动 —— 尤其训练内那行是 OPERATIONS §6.7:212 ⑤ 指定的「免费守卫」入口。

    mode='last'           取平台末端 = epoch 最大的**有效**中点（政策默认；确定性、不依赖 shadow）
    mode='argmax_capture2' 17.3.7~17.4.0 旧行为：val capture2 取 argmax（仅供复现旧 run）

    守卫⑤（免费）：val capture2 的 argmax 若恰落在**最后一个**有效中点上，说明曲线到末端仍在爬
      —— 很可能是早停把平台砍掉了 →「取平台末端」的前提不成立。**只告警、不改变选择**：
      选点规则本身必须是确定性的，不该被一个噪声级的 argmax 位移牵着走（那正是旧 argmax 选点的病根）。
    """
    valid = [(ep, s) for ep, s in pairs if not np.isnan(s)]
    notes = []
    if not valid:
        return None, notes
    last_ep = max(ep for ep, _ in valid)
    # 同分取最早（确定性；旧代码用严格 `>` 在字符串序里取到的是「最早」，数值序下保持一致）
    amax_ep = max(valid, key=lambda t: t[1])[0]
    if mode == 'argmax_capture2':
        chosen = amax_ep
        notes.append("  [选点] 规则=argmax(val capture2) —— 17.3.7~17.4.0 旧行为，仅供复现；"
                     "已判为「在不可分辨的差上取 argmax = 拟合噪声」")
    else:
        chosen = last_ep
        notes.append("  [选点] 规则=取平台末端（epoch 最大的**有效** midpoint）"
                     "—— 政策默认，确定性，不依赖 shadow")
    if amax_ep == last_ep:
        notes.append(f"  ⚠⚠ 守卫⑤：val capture2 的 argmax 落在**末端** ep{last_ep} —— 曲线到最后"
                     f"一个中点仍在爬，很可能是早停把平台砍掉了 →「取平台末端」前提不成立。"
                     f"请看上面逐中点那行曲线确认；必要时 RESUME 续训，或调 EARLYSTOP_CAP2_GUARD / "
                     f"CAP2_GUARD_MAX_EXTRA 重跑。")
    else:
        notes.append(f"  ✓ 守卫⑤：val capture2 的 argmax 在 ep{amax_ep}（末端 ep{last_ep} 之内）"
                     f"→ 平台已形成，规则成立")
    return chosen, notes


# ---------- 采样器选择（17.4.3：与 RANK_LOSS_W 解耦）----------
def use_grouped_sampler(rank_loss_w, mode='auto'):
    """是否用 GroupedBatchSampler（整组打包，保证 batch 内组内成对样本非空）。

    'auto' = 旧行为：Grouped ⟺ RANK_LOSS_W > 0（**默认，逐字不改现有一切 run**）
    '1'/'0'（及 true/false/yes/no/on/off）= 显式指定，与 rank_loss_w 无关。
    ⚠ 为何要能单独打开采样器：**默认路径**的原 sampler（`CircuitGroupSampler` 整电路打包）
      下 80 行 batch 的**零成对样本占比** full 92.1% / rest 49.0% / m4 26.7%（Grouped 0.0%）
      ⇒ 成对项大量为空 ⇒ 旧开关下「采样器效应」与「损失效应」分不开（§13.6 那次否证即如此）。
      ⚠ 17.4.4 更正：原记「随机 shuffle rest 94.0 / m4 67.3 / full 54.2」是**随机置换**的读数
        （只对应离群点清洗分支站点2 的 else），不是默认路径。见 _t_sampler_live.py Part C。
    """
    m = str(mode).strip().lower()
    if m in ('1', 'true', 'yes', 'on'):
        return True
    if m in ('0', 'false', 'no', 'off'):
        return False
    return float(rank_loss_w) > 0


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
            pred_log, _ = model(data.x, data.edge_index, data.batch, corner, csig, getattr(data, 'struct_prior', None))
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

def _pairwise_rank_loss(pred_log, target_log, grp):
    """成对排序损失：pred_log/target_log=(B,),grp=(B,)int组ID。
    同组内有序对(i,j):真实ti<tj时推pred_i<pred_j,hinge损失max(0,margin-(pred_j-pred_i))。"""
    if not torch.isfinite(pred_log).all() or not torch.isfinite(target_log).all():
        return torch.zeros((), device=pred_log.device)
    loss = torch.tensor(0.0, device=pred_log.device)
    n = 0
    for g in grp.unique():
        m = (grp == g)
        if m.sum() < 2:
            continue
        p = pred_log[m]; t = target_log[m]
        dp = p.unsqueeze(0) - p.unsqueeze(1)              # (k,k): dp[i,j]=p_i-p_j
        dt = t.unsqueeze(0) - t.unsqueeze(1)              # dt[i,j]=t_i-t_j
        viol = torch.relu(dp + RANK_MARGIN)               # p_i-p_j + margin >0 → 惩罚
        mask = dt < 0
        v = viol[mask]
        if v.numel() > 0:
            loss = loss + v.mean()
            n += 1
    return loss / max(n, 1)


def train_one_epoch(model, loader, optimizer, device, delta=1.0, show_progress=False, teacher_preds=None):
    model.train()
    total_loss = 0
    total_batches = len(loader)
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        corner = data.corner_cond.to(device) if hasattr(data, 'corner_cond') else None
        csig = data.circuit_sig.to(device) if hasattr(data, 'circuit_sig') else None
        out, _ = model(data.x, data.edge_index, data.batch, corner, csig, getattr(data, 'struct_prior', None))
        target_log = torch.log10(data.y + 1e-12)
        residual = out - target_log
        abs_res = torch.abs(residual)
        sample_loss = torch.where(abs_res <= delta,
                                  0.5 * residual ** 2,
                                  delta * (abs_res - 0.5 * delta))
        weights = torch.tensor([PIN_WEIGHTS.get(pin, 1.0) for pin in data.switching_pin], device=device)
        loss = (sample_loss * weights).mean()
        # 蒸馏项（KD）：软标签回归 + teacher 排序监督（teacher 预测按 row_idx 索引，log10 空间）
        if KD_ENABLED and teacher_preds is not None and hasattr(data, 'row_idx'):
            _tlog = torch.tensor(teacher_preds[data.row_idx.cpu().numpy()],
                                 dtype=torch.float, device=device)
            if KD_MODE in ('reg', 'reg+rank'):
                loss = loss + KD_LAMBDA * F.mse_loss(out, _tlog)
            if KD_MODE in ('rank', 'reg+rank') and hasattr(data, 'grp'):
                loss = loss + KD_RANK_W * _pairwise_rank_loss(out, _tlog, data.grp)
        # 组内成对排序损失：同组变体真实 t_i<t_j 时，推动 pred_i<pred_j（间隔 margin）
        if RANK_LOSS_W > 0 and hasattr(data, 'grp'):
            loss = loss + RANK_LOSS_W * _pairwise_rank_loss(out, target_log, data.grp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if show_progress and (i + 1) % 200 == 0:
            print(f"    batch {i+1}/{total_batches} ({100*(i+1)/total_batches:.0f}%)")
    return total_loss / len(loader)

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
            out, _ = model(data.x, data.edge_index, data.batch, corner, csig, getattr(data, 'struct_prior', None))
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
                      gb_hash + data_hash + STRUCT_MODE, "Graph cache")
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
        f"seed{TRAIN_SEED}",
        f"hdim{HIDDEN_DIM}",
        f"nlay{NUM_LAYERS}",
        f"struct{STRUCT_MODE}",
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

    # ---------- 数据集路径：V2（batch_v2_full + batch_v2_io）或旧 V1（delivery1+2） ----------
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import glob
    static_parquets = []
    dynamic_parquets = []
    if USE_V2:
        # V2 数据（默认 batch_v2_full + batch_v2_rest；DATA_BATCHES env 可覆盖；支持 *_partN.parquet）
        for batch in DATA_BATCHES.split(','):
            batch = batch.strip()
            if not batch:
                continue
            sp = os.path.join(data_dir, f"data/{batch}/circuit_static.parquet")
            dp = os.path.join(data_dir, f"data/{batch}/timing_arcs.parquet")
            if os.path.exists(sp) and os.path.exists(dp):
                static_parquets.append(sp); dynamic_parquets.append(dp)
                print(f"V2 data: {batch}")
            else:
                sparts = sorted(glob.glob(os.path.join(data_dir, f"data/{batch}/circuit_static_part*.parquet")))
                dparts = sorted(glob.glob(os.path.join(data_dir, f"data/{batch}/timing_arcs_part*.parquet")))
                if sparts and dparts:
                    static_parquets.extend(sparts); dynamic_parquets.extend(dparts)
                    print(f"V2 data: {batch} (parts)")
                else:
                    print(f"V2 data not found, skipping: {batch}")
        four_pin_only_eff = False   # rest/io 含任意 I/O，V2 下不做 4-pin 过滤
    else:
        # delivery1 + delivery2 合并（~54 万行，1,437 电路。旧数据在 archive_v13.1/）
        for prefix in ['data/delivery1', 'data/delivery2']:
            for b in ['batch1', 'batch2', 'batch3']:
                sp = os.path.join(data_dir, f"{prefix}/{b}/circuit_static.parquet")
                dp = os.path.join(data_dir, f"{prefix}/{b}/timing_arcs.parquet")
                if os.path.exists(sp) and os.path.exists(dp):
                    static_parquets.append(sp); dynamic_parquets.append(dp)
                    continue
                # delivery2/batch3 只有 part 文件，逐个加载
                sparts = sorted(glob.glob(os.path.join(data_dir, f"{prefix}/{b}/circuit_static_part*.parquet")))
                dparts = sorted(glob.glob(os.path.join(data_dir, f"{prefix}/{b}/timing_arcs_part*.parquet")))
                if sparts and dparts:
                    static_parquets.extend(sparts); dynamic_parquets.extend(dparts)
        four_pin_only_eff = FOUR_PIN_ONLY
    # 启动时检查：如果代码或数据变了，自动清除过期缓存
    check_and_clear_cache(static_parquets, dynamic_parquets)

    for p in static_parquets + dynamic_parquets:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    # 16.5.0 内存优化：wave OFF 时按列读取，排除 transistor_wave_json（占动态 df ~93% 内存，全量 7.76GB→0.54GB）
    # 16.11.12: USE_IDS_AVG_APPROX(近似 ids_avg) 也不需要 wave 列（特征由系数/GBDT 算，不读真实 wave）→ 同样排除，省 ~12GB/训练
    _skip_wave = (not USE_TRANSISTOR_WAVE) or \
        (str(os.environ.get('USE_IDS_AVG_APPROX', '0')) != '0')
    if _skip_wave:
        import pyarrow.parquet as _pq
        dynamic_dfs = []
        for _p in dynamic_parquets:
            _cols = [c for c in _pq.read_schema(_p).names if c != 'transistor_wave_json']
            dynamic_dfs.append(pd.read_parquet(_p, columns=_cols))
    else:
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

    # 16.4.0 内存修复：wave OFF 时 transistor_wave_json 列占动态 df ~93% 内存（实测 7.2GB/副本），直接剔除
    if not USE_TRANSISTOR_WAVE and 'transistor_wave_json' in dynamic_df.columns:
        dynamic_df = dynamic_df.drop(columns=['transistor_wave_json'])
        print("wave OFF: 剔除 transistor_wave_json 列（省 ~7GB/副本）")

    # ---------- 组大小过滤：剔除 < MIN_GROUP_SIZE 变体的组（单变体/小组无排序价值，2026-08-28）----------
    if MIN_GROUP_SIZE > 1 and 'expr' in dynamic_df.columns:
        gsize = dynamic_df.groupby('expr')['circuit_id'].nunique()
        keep_exprs = gsize[gsize >= MIN_GROUP_SIZE].index.astype(str)
        n_before = len(dynamic_df)
        dynamic_df = dynamic_df[dynamic_df['expr'].astype(str).isin(keep_exprs)]
        print(f"组大小过滤(>= {MIN_GROUP_SIZE} 变体): 样本 {n_before} -> {len(dynamic_df)}, "
              f"expr {len(gsize)} -> {len(keep_exprs)}")

    # ---------- 可选：只保留4引脚标准电路（在划分前过滤，避免空split；V2 下跳过）----------
    if four_pin_only_eff:
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
    # 按 expr 分组划分：同一 expr 的等价变体整组进同一 split（下游择优任务的正确切分，无泄漏）
    id_to_expr = (dict(zip(dynamic_df['circuit_id'].astype(str), dynamic_df['expr'].astype(str)))
                  if 'expr' in dynamic_df.columns else None)
    train_ids, val_ids, test_ids = split_by_expr(circuit_ids, id_to_expr, seed=SPLIT_SEED)
    print(f"划分(按expr, SPLIT_SEED={SPLIT_SEED}): train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} 电路")
    # 切分固定后，用 TRAIN_SEED 独立控制模型初始化/训练随机性（便于同切分多seed集成）
    set_seed(TRAIN_SEED)

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

    # ---------- Scaler 拟合（匹配 DelayDataset._get_dynamic_features 逻辑；V2 用每电路引脚 + JSON 列）----------
    # 每电路引脚（V2 任意 I/O 用；V1 回退全局 pins）
    def _cell_pins(cid):
        try:
            ip = json.loads(static_df.loc[cid, 'input_pins_json']) if isinstance(static_df.loc[cid, 'input_pins_json'], str) \
                else static_df.loc[cid, 'input_pins_json']
            if isinstance(ip, list) and ip:
                return list(ip)
        except Exception:
            pass
        return pins
    train_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)]
    all_cont_features = []
    for _, row in train_dynamic.iterrows():
        switching = row.get('switching_pin', '')
        global_slew = row.get('slew_s', 0.0)
        out_load = row.get('output_load_f', 0.0)
        loads_dict = pin_loads_map.get(row['circuit_id'], {})
        row_pins = _cell_pins(row['circuit_id'])
        # V2 per-pin JSON 列
        pin_slew, pin_load = {}, {}
        try:
            _v = json.loads(row.get('pin_slew_json')) if isinstance(row.get('pin_slew_json'), str) else row.get('pin_slew_json')
            if isinstance(_v, dict): pin_slew = {str(k): v for k, v in _v.items()}
        except Exception: pass
        try:
            _v = json.loads(row.get('pin_load_json')) if isinstance(row.get('pin_load_json'), str) else row.get('pin_load_json')
            if isinstance(_v, dict): pin_load = {str(k): v for k, v in _v.items()}
        except Exception: pass

        for pin in row_pins:
            # 匹配 data_loader 逻辑：优先 per-pin slew（JSON 或列），否则只有切换引脚用全局 slew
            if pin in pin_slew:
                slew_val = pin_slew[pin]
            else:
                slew_col = f'slew_{pin}'
                if slew_col in row.index and pd.notna(row[slew_col]):
                    slew_val = row[slew_col]
                elif pin == switching:
                    slew_val = global_slew
                else:
                    slew_val = 0.0
            # 匹配 data_loader 逻辑：优先 per-pin load（JSON 或列），否则静态字典
            if pin in pin_load:
                load_val = pin_load[pin]
            else:
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

    # V2 推荐结构模式（14.4.4 结论：logic_only 最优；在 DelayDataset 构建前生效）
    if USE_V2 and V2_STRUCT_MODE:
        import config as _cfg
        _cfg.STRUCT_MODE = V2_STRUCT_MODE
        print(f"[V2] STRUCT_MODE -> {V2_STRUCT_MODE}")

    # ---------- 创建数据集 ----------
    # 16.4.0 内存修复：一次性构建 train/val/test 子集并传入（替代三个 DelayDataset 各自全量重读 parquet；
    # wave 列 ~7GB/副本，此前每 run 持有 5 份 + 3 次重读堆残留 ≈ 41GB → 现仅持有 3 份子集 ≈ 7.6GB）
    val_df = dynamic_df[dynamic_df['circuit_id'].isin(val_ids)]
    test_df = dynamic_df[dynamic_df['circuit_id'].isin(test_ids)]
    # train_dynamic（scaler 拟合用）即 train 子集，直接复用，不再另建
    train_df = train_dynamic
    del dynamic_df
    import gc as _gc
    _gc.collect()
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)   # glibc 归还空闲堆页给 OS，降稳态 RSS（Linux）
    except Exception:
        pass
    train_dataset = DelayDataset(static_parquets, dynamic_parquets, train_ids, scaler, CACHE_DIR,
                                 dynamic_df=train_df, prefiltered=True)
    val_dataset = DelayDataset(static_parquets, dynamic_parquets, val_ids, scaler, CACHE_DIR,
                               dynamic_df=val_df, prefiltered=True)
    test_dataset = DelayDataset(static_parquets, dynamic_parquets, test_ids, scaler, CACHE_DIR,
                                dynamic_df=test_df, prefiltered=True)
    del train_df, val_df, test_df
    _gc.collect()
    print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # ---------- 蒸馏：加载 teacher 预测（KD_ENABLED 时；按 dataset row_idx 对齐） ----------
    _kd_teacher = None
    if KD_ENABLED and not KD_PREDS_ONLY:
        _kd_path = os.path.join(KD_TEACHER_DIR, 'kd_teacher_preds_train.npy')
        assert os.path.exists(_kd_path), f"[KD] 未找到 teacher 预测: {_kd_path}"
        _kd_teacher = np.load(_kd_path).astype(np.float32)
        assert len(_kd_teacher) == len(train_dataset), \
            f"[KD] teacher 预测长度 {len(_kd_teacher)} != train 样本数 {len(train_dataset)}"
        print(f"[KD] 已加载 teacher 预测: {_kd_path} ({len(_kd_teacher)} 行, KD_MODE={KD_MODE}, "
              f"λ={KD_LAMBDA}, κ={KD_RANK_W})")

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
    # 17.4.3：采样器由 use_grouped_sampler 决定，不再看 RANK_LOSS_W（'auto' 下行为与旧代码逐字相同）
    _grouped = use_grouped_sampler(RANK_LOSS_W, USE_GROUPED_SAMPLER)
    print(f"[采样器] RANK_LOSS_W={RANK_LOSS_W} USE_GROUPED_SAMPLER={USE_GROUPED_SAMPLER!r} → "
          f"{'GroupedBatchSampler(整组打包)' if _grouped else 'CircuitGroupSampler(原行为)'}")
    if _grouped:
        from src.utils import GroupedBatchSampler
        sampler = GroupedBatchSampler(train_dataset.group_ids, BATCH_SIZE, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=2)
    else:
        sampler = CircuitGroupSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    sample_data = next(iter(train_loader))
    in_dim = sample_data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Input dim: {in_dim}, Device: {device}")

    # ---------- 蒸馏：teacher 预测导出模式（KD_PREDS_ONLY=1，一次性，在 teacher 目录跑） ----------
    # 输出 kd_teacher_preds_{train,val,test}.npy（log10 空间，按下标 idx 对齐 student dataset）
    # 并对拍 test 排序指标（应与该 checkpoint 的 SUMMARY 一致），对拍通过再用于蒸馏训练。
    if KD_PREDS_ONLY:
        assert KD_TEACHER_CKPT, '[KD] KD_PREDS_ONLY=1 需要 KD_TEACHER_CKPT=<checkpoint 路径>'
        os.makedirs(KD_TEACHER_DIR, exist_ok=True)
        t_model = DelayGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                           dropout=DROPOUT, num_gate_types=num_gate_types,
                           gate_embed_dim=GATE_EMBED_DIM).to(device)
        t_model.load_state_dict(torch.load(KD_TEACHER_CKPT, map_location=device, weights_only=False))
        t_model.eval()
        for _name, _ds in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            _ld = DataLoader(_ds, batch_size=BATCH_SIZE, num_workers=2)
            _pl = []
            with torch.no_grad():
                for _d in _ld:
                    _d = _d.to(device)
                    _corner = _d.corner_cond.to(device) if hasattr(_d, 'corner_cond') else None
                    _csig = _d.circuit_sig.to(device) if hasattr(_d, 'circuit_sig') else None
                    _out, _ = t_model(_d.x, _d.edge_index, _d.batch, _corner, _csig,
                                      getattr(_d, 'struct_prior', None))
                    _pl.append(_out.cpu().numpy())
            _arr = np.concatenate(_pl).astype(np.float32)
            _p = os.path.join(KD_TEACHER_DIR, f'kd_teacher_preds_{_name}.npy')
            np.save(_p, _arr)
            print(f"[KD] saved {_p} ({len(_arr)} rows)")
        # 对拍校验：test 预测复算排序指标（应与 teacher SUMMARY 一致）
        _td = test_dataset.dynamic_df.reset_index(drop=True)
        _tp = np.clip(10 ** np.load(os.path.join(KD_TEACHER_DIR, 'kd_teacher_preds_test.npy')),
                      1e-12, 1e-8)
        _tt = test_dataset.dynamic_df['DELAY'].to_numpy(dtype=np.float64)
        _rk = ranking_metrics(_td, _tp, _tt, avg_delay=USE_V2)
        _hi = _rk.get('hi_spread', _rk)
        def _r2(x): return x.get('recall_at_k', {}).get(2, {}).get('strict', {}).get('hit_pct', 0.0)
        print(f"[KD] 对拍 test(全局): regret={_rk.get('regret_pct', float('nan')):.2f}% "
              f"sp={_rk.get('spearman', 0):.3f} cap={_rk.get('captured_pct', 0)*100:.1f}% "
              f"r2={_r2(_rk)*100:.1f}%")
        print(f"[KD] 对拍 test(spread>10%): regret={_hi.get('regret_pct', float('nan')):.2f}% "
              f"sp={_hi.get('spearman', 0):.3f} cap={_hi.get('captured_pct', 0)*100:.1f}% "
              f"r2={_r2(_hi)*100:.1f}%")
        print('[KD] 应与 teacher 的 SUMMARY 一致；对拍通过后再用于蒸馏训练。')
        return

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
        # 17.4.3：同站点 1，采样器不再由 RANK_LOSS_W 决定（'auto' 下行为与旧代码逐字相同）
        _grouped = use_grouped_sampler(RANK_LOSS_W, USE_GROUPED_SAMPLER)
        print(f"[采样器/清洗后] USE_GROUPED_SAMPLER={USE_GROUPED_SAMPLER!r} → "
              f"{'GroupedBatchSampler(整组打包)' if _grouped else '随机 shuffle(原行为)'}")
        if _grouped:
            from src.utils import GroupedBatchSampler
            sub_gids = [train_dataset.group_ids[i] for i in keep_indices]
            sampler = GroupedBatchSampler(sub_gids, BATCH_SIZE, shuffle=True)
            train_loader = DataLoader(train_subset, batch_sampler=sampler, num_workers=2)
        else:
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    else:
        print("\n跳过离群点清洗（未启用或样本量过少）\n")

    model = DelayGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
                     num_gate_types=num_gate_types,
                     gate_embed_dim=GATE_EMBED_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---------- eval-only 模式：不训练，直接从 checkpoint 重新生成 test_predictions.npz ----------
    # 用途：修复「SUMMARY 显示 midpoint 但 npz 存的是 best_model.pt」的不一致（已跑完的 seed 用 midpoint 重生成）
    # EVAL_ONLY 取值: 'midpoint'（自动按加权分数选最优 midpoint）| 'best'（best_model.pt）| 显式 checkpoint 路径
    eval_only = os.environ.get('EVAL_ONLY', '')
    if eval_only:
        import glob as _gb
        # 选点数据用 **val**（不是 test）：理由见下方训练内 midpoint 块顶部的 17.3.7 注释。
        sel_dyn = val_dataset.dynamic_df.reset_index(drop=True)
        if eval_only == 'midpoint':
            # ⚠ key= 数值序（同训练内那块）：sorted(glob()) 是字符串序，ep100 会排在 ep50 前
            mid_files = sorted(_gb.glob(os.path.join(OUTPUT_DIR, 'midpoint_ep*.pt')), key=midpoint_epoch_of)
            assert mid_files, f"[eval-only] 未找到 midpoint 文件于 {OUTPUT_DIR}"
            pairs = []
            for mf in mid_files:
                ep_num = midpoint_epoch_of(mf)
                if ep_num < 0:
                    continue
                model.load_state_dict(torch.load(mf, map_location=device, weights_only=False))
                _, _, mp_preds, mp_targets = evaluate(model, val_loader, device)
                mn = min(len(sel_dyn), len(mp_preds))
                mrk = ranking_metrics(sel_dyn.iloc[:mn], mp_preds[:mn], mp_targets[:mn], avg_delay=USE_V2)
                # 选点量 = 两阶段捕获率，取**全组**均值（不是 hi_spread 子集）。详见训练内 midpoint 块注释。
                score = mrk.get('capture2_pct', float('nan'))
                mhi = mrk.get('hi_spread', {})
                mrec = mhi.get('recall_at_k', {})
                mr2 = mrec.get(2, {}).get('strict', {}).get('hit_pct', 0.0)
                mr3 = mrec.get(3, {}).get('strict', {}).get('hit_pct', 0.0)
                # 其余量仅作诊断打印（不再进 score），便于与历史日志逐列对读
                print(f"[eval-only] ep{ep_num}(val): cap2={score:6.2f}%  "
                      f"| 诊断: regret={mhi.get('regret_pct', 100.0):.2f}% "
                      f"sp={mhi.get('spearman', 0.0):.3f} r2={mr2*100:.1f}% r3={mr3*100:.1f}%")
                # NaN = 该 epoch 在 val 上没有任何 m>=4 的组（不可统计）→ 不参与选点
                pairs.append((ep_num, score))
            # 与训练内那块**同一决策函数**（政策共用），避免两处选点规则各自漂移
            best_ep, sel_notes = pick_midpoint(pairs, MIDPOINT_SELECT)
            for _n in sel_notes:
                print(f"[eval-only]{_n}")
            if best_ep is None:
                # 这条路径是显式的诊断性重生成（会覆写 test_predictions.npz）→ 直接失败，
                # 绝不带着错误 ckpt 落盘 npz
                raise RuntimeError("[eval-only] 所有 midpoint 在 val 上的 capture2 均为 NaN（无 m>=4 组）→ 无法选点")
            ckpt = os.path.join(OUTPUT_DIR, f'midpoint_ep{best_ep}.pt')
            _rule = '取平台末端' if MIDPOINT_SELECT != 'argmax_capture2' else '选于 val(argmax)'
            print(f"[eval-only] best midpoint ep{best_ep} (capture2={dict(pairs)[best_ep]:.2f}%, {_rule}, "
                  f"MIDPOINT_SELECT={MIDPOINT_SELECT})")
        elif eval_only == 'best':
            ckpt = os.path.join(OUTPUT_DIR, 'best_model.pt')
        else:
            ckpt = eval_only
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        test_loss, test_rel_err, preds, targets = evaluate(model, test_loader, device)
        np.savez(os.path.join(OUTPUT_DIR, 'test_predictions.npz'), preds=preds, targets=targets)
        print(f"[eval-only] saved test_predictions.npz from {ckpt} (test_rel_err={test_rel_err:.2f}%)")
        return

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
    best_val_loss = float('inf')
    best_sel = float('inf')
    patience_counter = 0
    plateau_counter = 0
    val_err_history = []
    val_loss_history = []
    train_loss_history = []
    # 17.3.8：两阶段捕获率（val 上，逐 epoch）—— best_model 选点 / 早停守卫 / 日志共用
    val_dyn = val_dataset.dynamic_df.reset_index(drop=True)
    cap2_best, cap2_best_ep, cap2_guard_warned = -float('inf'), 0, False
    plateau_triggered = False
    last_lr = LEARNING_RATE
    lr_decayed = False
    # 排序选点初始化
    if BEST_RANK_METRIC != 'none':
        best_rank_val = float('inf') if BEST_RANK_METRIC == 'regret' else float('-inf')
        rank_model_path = os.path.join(OUTPUT_DIR, 'best_rank_model.pt')

    print("\nStart training...")
    t_train_start = time.time()
    # 16.11.12: RESUME_EPOCH 续训——加载 midpoint_ep{N}.pt 权重，从第 N epoch 继续
    # （optimizer/LR schedule 从头，权重保留；用于中断后从 checkpoint 续跑，省已训 epoch）
    start_epoch = 0
    _re = os.environ.get('RESUME_EPOCH', '')
    if _re:
        try:
            _re_n = int(_re)
            _re_path = os.path.join(OUTPUT_DIR, f'midpoint_ep{_re_n}.pt')
            if os.path.exists(_re_path):
                model.load_state_dict(torch.load(_re_path, map_location=device, weights_only=False))
                start_epoch = _re_n
                print(f"[resume] 已加载 {_re_path}，从 epoch {start_epoch} 续训")
            else:
                print(f"[resume] WARN: {_re_path} 不存在，从头训练")
        except (ValueError, Exception) as e:
            print(f"[resume] WARN: RESUME_EPOCH 解析失败 ({e})，从头训练")
    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, delta=HUBER_DELTA,
                                     teacher_preds=_kd_teacher)
        val_loss, val_rel_err, v_preds, v_targets = evaluate(model, val_loader, device)

        # 17.3.8：复用上面这次 val 前向的预测算两阶段捕获率（**不额外前向**；实测单次
        # 0.09~1.25s，取决于每组的变体数 V，整趟 300 epoch 累计 <6 分钟，可忽略）。
        # 失败一律按 NaN 处理，绝不因为一个诊断量中断已跑了几小时的训练。
        val_cap2 = float('nan')
        try:
            _mn = min(len(val_dyn), len(v_preds))
            val_cap2 = ranking_metrics(val_dyn.iloc[:_mn], v_preds[:_mn], v_targets[:_mn],
                                       avg_delay=USE_V2).get('capture2_pct', float('nan'))
        except Exception as _e:
            print(f"  [cap2] 计算失败（不中断训练，按 NaN 处理）: {_e}")
        if np.isfinite(val_cap2) and val_cap2 > cap2_best:
            cap2_best, cap2_best_ep = val_cap2, epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Rel Err: {val_rel_err:.2f}% | Cap2: {val_cap2:6.2f}%")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        val_err_history.append(val_rel_err)
        val_loss_history.append(val_loss)
        train_loss_history.append(train_loss)

        # 检测 LR 是否已衰减（LR 降低是突破平台期的契机，在此之前不早停）
        if current_lr < last_lr * 0.99:
            lr_decayed = True
        last_lr = current_lr

        # 报告指标：始终追踪最小 val_rel_err
        if val_rel_err < best_val_rel:
            best_val_rel = val_rel_err
        # 保存 checkpoint：按 config.BEST_MODEL_METRIC 选点（当前 capture2；可选
        # smoothed_rel_err / val_loss / val_rel_err —— 见 config.py 的 17.3.8 注释）
        if BEST_MODEL_METRIC == 'capture2':
            # 取负 → 沿用下面 sel<best_sel 的「越小越好」。非有限时记 inf → 该 epoch 不落盘：
            # **绝不用 NaN 去比大小**（NaN 比较恒 False，会让 best_model.pt 一个 epoch 都不写）
            sel = -val_cap2 if np.isfinite(val_cap2) else float('inf')
        elif BEST_MODEL_METRIC == 'val_loss':
            sel = val_loss
        elif BEST_MODEL_METRIC == 'smoothed_rel_err':
            sel = float(np.mean(val_err_history[-BEST_SMOOTH_WINDOW:]))
        else:
            sel = val_rel_err
        if sel < best_sel:
            best_sel = sel
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            _sel_repr = (f"capture2={-sel:.2f}%" if BEST_MODEL_METRIC == 'capture2'
                         else f"{BEST_MODEL_METRIC}={sel:.4f}")
            print(f"  >>> New best model saved ({_sel_repr}, ValRelErr={val_rel_err:.2f}%)")

        # 中途快照（不影响训练 RNG，只写文件）
        if SAVE_MIDPOINTS and (epoch + 1) % MIDPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'midpoint_ep{epoch+1}.pt'))

        # 排序选点：每隔 N epoch 在 val 上评估排序，按 BEST_RANK_METRIC 保存最优 checkpoint
        if BEST_RANK_METRIC != 'none' and (epoch + 1) % RANK_EVAL_INTERVAL == 0:
            model.eval()
            rp, rt = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    corner = data.corner_cond.to(device) if hasattr(data, 'corner_cond') else None
                    csig = data.circuit_sig.to(device) if hasattr(data, 'circuit_sig') else None
                    out, _ = model(data.x, data.edge_index, data.batch, corner, csig, getattr(data, 'struct_prior', None))
                    rp.append(out.cpu().numpy()); rt.append(data.y.cpu().numpy())
            model.train()
            try:
                rk = ranking_metrics(val_dataset.dynamic_df.reset_index(drop=True),
                                     np.concatenate(rp), np.concatenate(rt), avg_delay=USE_V2)
                vr = rk['regret_pct'] if BEST_RANK_METRIC == 'regret' else rk['spearman']
                better = (vr < best_rank_val) if BEST_RANK_METRIC == 'regret' else (vr > best_rank_val)
                if not np.isnan(vr) and better:
                    best_rank_val = vr
                    torch.save(model.state_dict(), rank_model_path)
                    print(f"  >>> Rank checkpoint saved @ ep{epoch+1} ({BEST_RANK_METRIC}={vr:.4f})")
            except Exception:
                pass

        # 早停判据：改用稳定的 val_loss。val_rel_err 被极端 corner 主导、逐 epoch 剧烈震荡，
        # 会在 val_loss 仍在下降时误判过拟合、提前砍断训练（10.3.3 即在 val_loss 0.0227→0.0200
        # 仍在降时被 rel err 噪声于 ep106 误停）。故计数器与平台检测一律以 val_loss 为准。
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            plateau_counter = 0
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

            # ---- 17.3.8 早停守卫：val_loss 已平台，但 capture2 仍在刷新 → 先不停 ----
            # 位置：放在两个早停出口**之前**，两个出口都要过这一关（否则守卫会被另一个绕过）。
            # 硬上限用 patience_counter 表达：它随 val_loss 不改善而单调增长，故
            # patience_counter < PATIENCE + CAP2_GUARD_MAX_EXTRA 就是「最多再多跑 MAX_EXTRA 个
            # epoch」，不需要额外的计数器。
            cap2_guard_blocks = (
                EARLYSTOP_CAP2_GUARD
                and np.isfinite(val_cap2)
                and (epoch + 1 - cap2_best_ep) < CAP2_GUARD_PATIENCE
                and patience_counter < PATIENCE + CAP2_GUARD_MAX_EXTRA
            )
            if cap2_guard_blocks and not cap2_guard_warned:
                cap2_guard_warned = True
                print(f"  [早停守卫] val_loss 已平台，但 capture2 于 ep{cap2_best_ep} 仍刷新"
                      f"（{cap2_best:.2f}%）→ 继续训练，最多再延 {CAP2_GUARD_MAX_EXTRA} epoch")

            # ---- 智能早停：LR 已衰减 + train 还在降 + val_loss 也停止下降 → 真过拟合 ----
            # plateau_counter>=PLATEAU_WINDOW 已保证 val_loss 连续 PLATEAU_WINDOW 个 epoch 无新低，
            # 不再看噪声大的 val_rel_err。
            if (plateau_counter >= PLATEAU_WINDOW
                    and epoch + 1 >= PLATEAU_MIN_EPOCHS
                    and lr_decayed
                    and not plateau_triggered
                    and not cap2_guard_blocks):
                recent_train = train_loss_history[-PLATEAU_WINDOW:]
                prev_start = max(0, len(train_loss_history) - 2 * PLATEAU_WINDOW)
                prev_train = train_loss_history[prev_start:len(train_loss_history) - PLATEAU_WINDOW]
                train_mean_recent = np.mean(recent_train)
                train_mean_prev = np.mean(prev_train) if prev_train else train_mean_recent
                # train loss 在持续下降（比前一窗口明显更低）而 val_loss 已停滞 → 过拟合
                train_still_improving = train_mean_recent < train_mean_prev - 0.0005

                if train_still_improving:
                    plateau_triggered = True
                    print(f"  >>> Plateau detected: train still improving ({train_mean_prev:.4f}→{train_mean_recent:.4f}) "
                          f"but val_loss stalled {plateau_counter} epochs (best_val_loss={best_val_loss:.4f}), stop.")
                    break

            if patience_counter >= PATIENCE and not cap2_guard_blocks:
                print(f"Early stopping (val_loss 连续 {PATIENCE} epoch 无改善)")
                break

    # 排序选点优先；fallback 回原 best_model
    # ⚠ 17.4.1（OPERATIONS §6.7:212）：**best_model.pt 不进部署候选**。此处装载它只为
    #   ① 下面的 val/test 基线与 per-corner / per-batch 分解打印；② 万一下面**没有任何可用
    #   midpoint** 时的退化兜底。真正的部署候选由下方 midpoint 块按政策（MIDPOINT_SELECT）选出。
    rp = os.path.join(OUTPUT_DIR, 'best_rank_model.pt')
    _bm = os.path.join(OUTPUT_DIR, 'best_model.pt')
    if BEST_RANK_METRIC != 'none' and os.path.exists(rp):
        model.load_state_dict(torch.load(rp))
    else:
        if not os.path.exists(_bm):
            # 17.3.8 兜底：BEST_MODEL_METRIC='capture2' 时若整趟 capture2 都非有限（例如 val 上
            # 无 m>=4 的组，或服务器上的 src/utils.py 还没带上 capture2_pct 这个 key），sel 恒为
            # inf → 一个 epoch 都没落盘 → 原先这里 torch.load 会直接崩，把已训完的权重全废掉。
            # 明确报出来 + 用当前权重兜住，让整趟 run 的产物（midpoint/SUMMARY）仍可用。
            torch.save(model.state_dict(), _bm)
            print(f"  ⚠ best_model.pt 缺失（BEST_MODEL_METRIC={BEST_MODEL_METRIC} 全程无有效值）"
                  f" → 已用当前 epoch 权重兜底落盘；选点请以 midpoint 为准")
        model.load_state_dict(torch.load(_bm))
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
                    abs_err_ps = np.mean(np.abs(preds[mask] - targets[mask])) * 1e12
                    delay_ps = np.mean(targets[mask]) * 1e12
                    print(f"  {c}: n={mask.sum():,}  mean_err={np.mean(err):.1f}%  "
                          f"abs_err={abs_err_ps:.2f}ps  mean_delay={delay_ps:.2f}ps")
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
        if USE_V2:
            # V2：expr8000~8999 = batch_v2_full(4-pin)，expr9000+ = batch_v2_io(任意I/O)
            b1_mask = (expr_nums >= 8000) & (expr_nums <= 8999)
            b2_mask = expr_nums >= 9000
            b3_mask = np.zeros_like(expr_nums, dtype=bool)
            batch_labels = [('V2-full(4pin)', b1_mask), ('V2-io(任意IO)', b2_mask)]
        else:
            b1_mask = (expr_nums >= 0) & (expr_nums <= 199)     # batch1 + batch1b
            b2_mask = (expr_nums >= 200) & (expr_nums <= 999)   # batch2
            b3_mask = expr_nums >= 1000                           # batch3
            batch_labels = [('B1(全sweep)', b1_mask), ('B2(稀疏)', b2_mask), ('B3(新建)', b3_mask)]
        for label, mask in batch_labels:
            if mask.sum() > 0:
                err = np.abs(preds[mask] - targets[mask]) / targets[mask] * 100
                print(f"  {label}: n={mask.sum():,}  mean_err={np.mean(err):.1f}%")

    # ---------- 中途快照回溯（训练已完成，不影响 RNG）----------
    # 17.3.7：选点在 **val** 上做（原来在 test 上）。理由——test 是整条链上唯一没参与过任何
    #   决策的数据；一旦用它挑 epoch，它就不再是测试集，报出的数变成「N 个候选里最好的那个
    #   在 test 上的成绩」，系统性偏乐观，偏量约等于 epoch 间噪声（实测臂内 1.6~2.7pp，
    #   是臂间差异 0.46pp 的数倍），足以把臂的名次排反。val 本就在承担选点职责
    #   （best_model.pt 按 config.BEST_MODEL_METRIC，**当前为 capture2**——17.3.8 起换的，
    #    此处旧注释误记为 smoothed_rel_err，17.4.1 更正；早停按 val_loss），
    #   多一项用途不新增污染；且 val 与
    #   test 同为 15% 的 expr 组、规模基本相同，判据噪声不变。选中的 ep 仍**在 test 上报数**
    #   （本块末尾那次 evaluate 不动）→ test 从此才是诚实的成绩单。
    #   分数公式本次**不改**（判据换成两阶段捕获率另出一版，两件事各自可单独验证）。
    #   17.4.1：**选点规则**改了 —— 从「val capture2 取 argmax」改为政策性的「取平台末端」，
    #   即 MIDPOINT_SELECT='last'（config.py 有原文与理由）。逐中点那行 `ep N: cap2=...` 的
    #   **格式逐字未动**：它是 OPERATIONS §6.7:212 ⑤ 指定的免费守卫入口（看 argmax 是否落在
    #   末端 → 判断早停有没有把平台砍掉）。上面的 val 侧评估保留为**诊断量**，不再决定选谁。
    orig_preds, orig_targets, orig_test_rel = preds.copy(), targets.copy(), test_rel_err
    mid_epoch = None
    if SAVE_MIDPOINTS:
        import glob as _gb
        # ⚠ key= 数值序：sorted(glob()) 是字符串序（ep100 会排在 ep50 前）——「取末端」必须数值序
        mid_files = sorted(_gb.glob(os.path.join(OUTPUT_DIR, 'midpoint_ep*.pt')), key=midpoint_epoch_of)
        if mid_files:
            # 注意：不能用外层的 test_dyn —— 它下面还要给 SUMMARY/per-corner/per-batch 用
            sel_dyn = val_dataset.dynamic_df.reset_index(drop=True)
            # 17.4.1：抬头原写 "weighted score, 选于 val" —— weighted score 自 17.3.7 已弃用，
            #   且选点规则已改为政策性的「取平台末端」（不再是在 val 上取 argmax）→ 两处都改掉，
            #   免得读日志的人以为这行还是旧的加权分选点。
            print("\n------- Midpoint Comparison (逐中点诊断；选点规则见下方 [选点] 行) -------")
            pairs = []
            for mf in mid_files:
                ep_num = midpoint_epoch_of(mf)
                if ep_num < 0:
                    continue
                model.load_state_dict(torch.load(mf, map_location=device, weights_only=False))
                _, _, mp_preds, mp_targets = evaluate(model, val_loader, device)
                mn = min(len(sel_dyn), len(mp_preds))
                mrk = ranking_metrics(sel_dyn.iloc[:mn], mp_preds[:mn], mp_targets[:mn], avg_delay=USE_V2)
                # 选点量 = 两阶段捕获率，取**全组**均值（不是 hi_spread 子集）。
                # 17.3.7 换判据：原 score（100·r3+50·r2+0.3·sp−0.2·regret+0.1·cap，recall@3 主导）
                #   弃用 —— 实测该 score 序与 Rust 部署序 Spearman = −1（42b 五点完全反序）；
                #   且 recall@3 是离散量、hi_spread 分档本是「遗憾未按 spread 归一」时代的补丁。
                # 两阶段捕获率 = (真最差 − 前3内真最优) / (真最差 − 真最优)：分母 = 可改进空间
                #   → 随 spread 归一、跨组可比，且逐字对应预留管线「GNN 前3 → SPICE 精排 → 取前3内真最优」。
                # ⚠ 与两阶段遗憾单调对应：regret_2stage = (1 − capture2) × spread → 勿两者并用。
                score = mrk.get('capture2_pct', float('nan'))
                mhi = mrk.get('hi_spread', {})
                mr = mhi.get('regret_pct', 100.0)
                ms = mhi.get('spearman', 0.0)
                mc = mhi.get('captured_pct', 0.0)
                mrec = mhi.get('recall_at_k', {})
                mr2 = mrec.get(2, {}).get('strict', {}).get('hit_pct', 0.0)
                mr3 = mrec.get(3, {}).get('strict', {}).get('hit_pct', 0.0)
                # 其余量仅作诊断打印（不再进 score），便于与历史日志逐列对读
                print(f"  ep{ep_num:>4d}: cap2={score:6.2f}%  | 诊断: regret={mr:.2f}% sp={ms:.3f} "
                      f"cap={mc:.1f}% r2={mr2*100:.1f}% r3={mr3*100:.1f}%")
                # NaN = 该 epoch 在 val 上没有任何 m>=4 的组（不可统计）→ 不参与选点
                pairs.append((ep_num, score))
            # ---- 按政策选点（决策与 EVAL_ONLY=midpoint 共用同一函数，防两处走偏）----
            best_ep, sel_notes = pick_midpoint(pairs, MIDPOINT_SELECT)
            for _n in sel_notes:
                print(_n)
            if best_ep is None:
                # 全部 epoch 不可统计 → 明确喊出来，不要静默退回（历史踩过「SUMMARY 显示 midpoint
                # 而 npz 存的是 best_model.pt」的不一致，别让它以新形式重演）。此处不抛异常：
                # 模型已训完、checkpoint 均已落盘，抛了会连带丢掉 SUMMARY。
                # ⚠ 这里退回 best_model.pt 是**退化路径**（没有任何可用 midpoint），与政策「best_model.pt
                #   不进部署候选」不矛盾：政策禁的是「有 midpoint 却去选 best_model」，不是「无 midpoint 时崩掉」。
                print("  ⚠ 所有 midpoint 的 val capture2 均为 NaN（无 m>=4 组）→ 未选点，沿用 best_model.pt")
                print("  ⚠⚠ 这是退化路径：本次 run **没有**合法部署候选，不要拿 best_model.pt 去 serve")
            else:
                best_score = dict(pairs)[best_ep]
                _rule = '取平台末端' if MIDPOINT_SELECT != 'argmax_capture2' else '选于 val(argmax)'
                print(f"  >>> Best midpoint: ep{best_ep} (capture2={best_score:.2f}%, {_rule}, "
                      f"MIDPOINT_SELECT={MIDPOINT_SELECT})")
                print(f"  ✓ 部署候选 = midpoint_ep{best_ep}.pt；best_model.pt 不进部署候选"
                      f"（OPERATIONS §6.7:212）")
                # 诚实标注：上面 per-corner / per-batch 两节消费的是本块**之前**装载的 best_model.pt
                # 权重（见本块上方那个装载点），自本行起（含 test_predictions.npz）才换成选中的 midpoint。
                print("  ⚠ 上方 per-corner / per-batch 各行来自 best_model.pt，不是本 midpoint —— 读 SUMMARY 时勿混。")
                mid_epoch = best_ep
                model.load_state_dict(torch.load(
                    os.path.join(OUTPUT_DIR, f'midpoint_ep{best_ep}.pt'),
                    map_location=device, weights_only=False))
                test_loss, test_rel_err, preds, targets = evaluate(model, test_loader, device)
                # 让 npz 与 SUMMARY 一致（midpoint 选点后的预测），否则集成脚本读到的是 best_model.pt
                np.savez(os.path.join(OUTPUT_DIR, 'test_predictions.npz'), preds=preds, targets=targets)

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
    if epoch + 1 >= EPOCHS:
        stop_reason = 'max_epochs(跑满)'
    elif plateau_triggered:
        stop_reason = 'plateau(train降但val_loss平→过拟合)'
    else:
        stop_reason = 'early_stop(val_loss连续无改善)'
    print(f"  Config: LR={LEARNING_RATE} LR_MIN={LR_MIN} LR_FACTOR={LR_FACTOR} HUBER={HUBER_DELTA} "
          f"BATCH={BATCH_SIZE} BEST_METRIC={BEST_MODEL_METRIC} SPLIT_SEED={SPLIT_SEED} TRAIN_SEED={TRAIN_SEED}")
    print(f"  停止: {stop_reason} @ epoch {epoch + 1}  (Best Val Rel Err {best_val_rel:.2f}%)")
    if mid_epoch is not None:
        print(f"  [显示的是 midpoint ep{mid_epoch} 的指标（若SAVE_MIDPOINTS=midpoint最优epoch已载入）]")
    # ---- 点精度（目标: 越小越好, 理想0）----
    print(f"  Test Median Rel Err: {float(np.median(np.abs(preds - targets) / targets)) * 100:.2f}%(→0)   "
          f"Mean Abs Err: {float(np.mean(np.abs(preds - targets))) * 1e12:.2f}ps(→0)   "
          f"(Mean Rel Err {test_rel_err:.2f}% ← 被小延迟放大,仅参考)")
    # ---- 排序（真实任务：等价变体择优。目标: Spearman→1, 遗憾→0%, top1/捕获/成对分辨→100%）----
    try:
        if len(test_dyn) == len(preds):
            rk = ranking_metrics(test_dyn, preds, targets, avg_delay=USE_V2)
            print(f"  [排序] 组(>=2)={rk['n_groups']}  Spearman={rk['spearman']:.3f}(→1)  "
                  f"选择遗憾={rk['regret_pct']:.2f}%(→0)  top1={rk['top1_acc']*100:.1f}%(→100)  "
                  f"捕获率={rk['captured_pct']:.1f}%(→100)  变体差中位={rk['spread_pct']:.1f}%")
            pa = rk['pair_acc']
            hi = rk.get('hi_spread', {})
            if hi.get('n', 0) > 0:
                print(f"  [排序 spread>10%] 组(>=2)={hi['n']}  Spearman={hi['spearman']:.3f}(→1)  "
                      f"选择遗憾={hi['regret_pct']:.2f}%(→0)  top1={hi['top1_acc']*100:.1f}%(→100)  "
                      f"捕获率={hi['captured_pct']:.1f}%(→100)")
            # recall@K：A=真#1进前K / B=前K有真前K之一，仅统计非平凡组(size>=K+1)
            def _fmt_recall(rec):
                return '  '.join(
                    f"@K={K} A={rec[K]['strict']['hit_pct']*100:.1f}%(n={rec[K]['strict']['n']}) "
                    f"B={rec[K]['lenient']['hit_pct']*100:.1f}%(n={rec[K]['lenient']['n']})"
                    for K in sorted(rec))
            print(f"  [recall@K 全局] {_fmt_recall(rk.get('recall_at_k', {}))}")
            if hi.get('recall_at_k'):
                print(f"  [recall@K spread>10%] {_fmt_recall(hi.get('recall_at_k', {}))}")
            print("  [成对分辨(按真实延迟差,→100%; <2%那档是贪心细粒度重写的关键)] " + "  ".join(
                f"{lab}:{pa[lab][0]:.0f}%(n={pa[lab][1]})" for lab in ['<2%', '2-5%', '5-10%', '>10%']))
    except Exception as _e:
        print(f"  [排序] 计算失败: {_e}")
    # 批次误差（安全的，处理 expr 不存在的情况）
    if 'expr' in test_dyn.columns and 'batch_labels' in dir():
        for label, mask in batch_labels:
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
