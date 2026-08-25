# 蒸馏方案：teacher 有 wave → student 无 wave（Rust 集成用）

> 决策依据（PROJECT_LOG 14.4）：v2nowave 实测 regret 8.07%/8.25%，比 v2wave（0.12%/0.48%）掉 ~8pt，远超 5% 阈值 → 蒸馏触发。
> 本方案全部落在 Python 训练侧；**不改 Rust、推理零仿真**，student 喂给 Rust 的特征与 nowave 完全一致。

---

## 0. 目标与判定标准

| 量 | 数值 | 说明 |
|---|---|---|
| 上界 | teacher v2wave123：regret 0.12%（test rel err 11.9%） | wave 可用时 |
| 下界 | v2nowave123：regret 8.25%（test rel err 31.3%） | 无 wave、无蒸馏 |
| **成功线** | **student 蒸馏后 regret < 3%**（目标 < 2%） | midpoint 最优 epoch 口径 |
| 硬约束 | 不改 Rust / 推理零仿真 / 特征 = nowave 同集 | Rust 集成即插即用 |

**边界（必须提前说清）**：student 输入没有 wave，只能学「静态特征与 wave 结果的相关性」，**不可能追平 0.12%**。本方案的现实目标是把 regret 从 8% 拉回 3% 以内。若 A1/A2 连 6% 都进不了，说明静态特征里 wave 信号不足，需走第 5 节兜底。

---

## 1. 总体架构

- **Teacher**：`v2wave123` 的 best midpoint checkpoint（`~/project-107-v2wave123/outputs/midpoint_ep150.pt`，score 46.36 最优）。可选升级：teacher 集成 = v2wave42 + v2wave123 预测平均（先不启用，作为 Phase B 变量）。
- **Student**：同一个 `DelayGNN`，仅 `USE_TRANSISTOR_WAVE=False`。**模型结构零改动**——`in_dim` 由 `sample_data.x.shape[1]` 自动推导（wave=3 个节点特征，关掉自动少 3 维）。
- **数据**：同一 V2 数据（batch_v2_full + batch_v2_io）+ 同一 `SPLIT_SEED=42` 切分 → **行对齐天然成立**：`DelayDataset.__getitem__(idx)` 按 `dynamic_df.iloc[idx]` 确定性构建，teacher/student 的 dataset row idx 一一对应，teacher 预测数组可直接按下标索引。

---

## 2. 三个组件

### 2.1 组件 A：预计算 teacher 预测（一次性，teacher 目录里跑）

用 `train_sweep.py` 内置的 **`KD_PREDS_ONLY` 导出模式**（不另写脚本，直接复用 main() 的切分/缩放/数据集构建，零复制、零漂移）：

```bash
# 在 ~/project-107-v2wave123 里执行（USE_TRANSISTOR_WAVE=1 默认，切分 SPLIT_SEED=42 与 student 一致）
KD_PREDS_ONLY=1 KD_TEACHER_CKPT=outputs/midpoint_ep150.pt KD_TEACHER_DIR=outputs \
  ~/venv/bin/python3 -u main.py
```

该模式复用 `train_sweep.main()` 的构建段（parquet 装配 + `split_by_expr(seed=SPLIT_SEED)` + scaler 拟合 + `DelayDataset`），加载指定 checkpoint 后对 train/val/test 三个**顺序 loader**（shuffle=False，保证行序 = dynamic_df 下标序）各跑一遍，收集 **log10 空间**的 `out`，保存：

- `KD_TEACHER_DIR/kd_teacher_preds_{train,val,test}.npy`（float32，长度 = 对应 dataset 行数，**按下标 idx 对齐**）

并**内置对拍校验**：用 `ranking_metrics(test, preds, targets, avg_delay=True)` 复算 test 排序指标并打印，应与 v2wave123 SUMMARY 一致（regret≈0.12% 等）——对拍通过才证明 npz 与行序对齐正确。

> 只训练用 train 的预测；val/test 的 npz 仅用于对拍校验（val 必须保持干净，早停/midpoint 选点不受污染）。

### 2.2 组件 B：student 训练接入蒸馏损失

**config.py 新增（全部 env 可覆盖，兼容 `setup_exp.sh` 模式）：**
```python
KD_ENABLED     = os.environ.get('KD_ENABLED', '0') == '1'
KD_TEACHER_DIR = os.environ.get('KD_TEACHER_DIR', '')   # kd_teacher_preds_*.npy 所在目录
KD_LAMBDA      = float(os.environ.get('KD_LAMBDA', '1.0'))  # 软标签回归权重
KD_RANK_W      = float(os.environ.get('KD_RANK_W', '1.0'))  # teacher 排序损失权重
KD_MODE        = os.environ.get('KD_MODE', 'reg+rank')      # 'reg' | 'rank' | 'reg+rank'
```

**src/data_loader.py**：`__getitem__` 末尾（`data.grp` 旁边，L525）加一行：
```python
data.row_idx = torch.tensor([idx], dtype=torch.long)
```
（向后兼容；离群清洗的 `Subset` 保留原 idx，`row_idx` 依旧指向 teacher npz 的正确行。）

**src/train_sweep.py**：
1. 数据集构建后（L441 附近）：`KD_ENABLED` 时加载 `np.load(KD_TEACHER_DIR/kd_teacher_preds_train.npy)`，并断言长度 == `len(train_dataset)`（防错位）。
2. `train_one_epoch`（L83~94）内、`loss` 计算后追加：
```python
if KD_ENABLED and _kd_teacher is not None and hasattr(data, 'row_idx'):
    tlog = torch.tensor(_kd_teacher[data.row_idx.cpu().numpy()], dtype=torch.float, device=device)
    if KD_MODE in ('reg', 'reg+rank'):
        loss = loss + KD_LAMBDA * F.mse_loss(out, tlog)                    # 软标签回归（log10 空间）
    if KD_MODE in ('rank', 'reg+rank') and hasattr(data, 'grp'):
        loss = loss + KD_RANK_W * _pairwise_rank_loss(out, tlog, data.grp)  # teacher 排序监督
```
> 关键复用：现有 `_pairwise_rank_loss(pred_log, target_log, grp)` 就是「按参考向量排序的 hinge 损失」——把 **teacher 的 log10 预测当 target_log 传入**，即得 teacher 引导的组内排序监督，**零新损失代码**。

**损失形态**：`huber(真值)` + `λ·MSE(student, teacher_log10)` + `κ·teacher_pairwise_rank`。**必须保留真值 huber 项**——Rust 贪心需要绝对延迟刻度，纯序蒸馏会丢刻度。

### 2.3 组件 C：setup_exp.sh 变体

```bash
# 与 v2nowave 相同 sed（关 wave）+ KD 环境变量
case "$V" in
  v2kd[0-9]*)
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${V#v2kd}/" config.py
    sed -i "s/^CACHE_DIR = .*/CACHE_DIR = \"cache107$V\"/" config.py ;;
esac
# 启动时追加：KD_ENABLED=1 KD_TEACHER_DIR=... KD_LAMBDA=... KD_MODE=... \
```
命名约定：`v2kd<teacher><mode><seed>`，如 `v2kd123reg42` = teacher=wave123、KD_MODE=reg、seed=42；`v2kd123rr42` = reg+rank。**student 跑之前先把 teacher 的 3 个 npz 拷到 student 目录（或 KD_TEACHER_DIR 指向共享路径）。**

---

## 3. 实验矩阵（按顺序执行）

### Phase A — 可行性（1 个 GPU 日，先出结论）
| run | 配置 | 判读 |
|---|---|---|
| A1 | `v2kd123reg42`（λ=1.0, reg） | 纯软标签回归能恢复到多少 |
| A2 | `v2kd123rr42`（λ=1.0, κ=1.0, reg+rank） | +teacher 排序监督是否再加分 |
| 对照 | v2nowave42（已有 8.07%）/ v2wave123 teacher（已有 0.12%） | 上/下界 |

**通过线**：A1 或 A2 的 midpoint regret < 4~5% → 进 Phase B；若都 > 6% → 静态特征 wave 信息不足，转第 5 节兜底。

### Phase B — 调参
- λ ∈ {0.5, 2.0}；κ ∈ {0.5, 2.0}（reg+rank 模式）；
- teacher 选择：wave123 单 teacher vs wave42+123 集成预测；
- 若用 softmax 排序蒸馏（可选变体）：温度 τ ∈ {1.0, 2.0, 5.0}，KL 替代 hinge（先不默认启用，hinge 版优先）。

### Phase C — 确认与交付
- 最优配置 × seed {42, 123}，2 seed；
- 与 v2nowave 集成做对照，写入结果记录（PROJECT_LOG）；
- 交付物：student checkpoint + test_predictions.npz；serve.py 喂无 wave 特征 → **Rust 侧零改动**。

---

## 4. 评估口径（与现有完全一致，不许改）

- `ranking_metrics(..., avg_delay=USE_V2)`，全局 + spread>10% 两档；
- midpoint 选点沿用现有 score：`-regret + 0.3·recall@2(A) + 0.3·spearman + 0.1·cap + 0.1·recall@3(A)`；
- 报告三件套：**teacher（上界）/ nowave（下界）/ kd-student**，同一 test 集。

---

## 5. 风险与兜底

| 风险 | 信号 | 兜底 |
|---|---|---|
| 静态特征里 wave 信号太少 | A1/A2 regret 仍 >6% | GNN_RUST_DATA_DIFF 5.6 阶段2（Rust 轻量 wave 估算），或接受 nowave + 窗口过滤（≤4pin） |
| 行对齐错位 | 对拍 test 指标 ≠ SUMMARY | 脚本内置长度断言 + 对拍打印；对齐错则重生成 npz |
| 离群清洗破坏索引 | A1 打印清洗前后条数 | `Subset` 保留原 idx，`row_idx` 仍有效（打印确认） |
| teacher 自身带噪（ids_charge P5 残留） | 蒸馏收益小 | 换 teacher 集成（wave42+123 平均）平滑 |
| 蒸馏过拟合 teacher 噪声 | 训练 loss 降但 regret 不降 | λ 网格 + midpoint 选点兜底 |

---

## 6. 代码改动清单

| 文件 | 改动 | 规模 |
|---|---|---|
| `config.py` | +7 个 KD_* 配置（env 可覆盖） | ~8 行 |
| `src/data_loader.py` | `__getitem__` 加 `data.row_idx`（L525 旁） | 1 行 |
| `src/train_sweep.py` | ①`KD_PREDS_ONLY` 导出模式（替代独立脚本，复用 main 的数据构建+对拍）；②加载 teacher npz；③`train_one_epoch` 加 2 个蒸馏项；④主循环传 `teacher_preds` | ~60 行 |
| `setup_exp.sh` | +v2kd* 变体分支（名称编码 teacher/mode/seed）+ 启动时注入 `KD_TEACHER_DIR` | ~18 行 |

## 7. 执行步骤

```bash
# ① teacher 侧：导出预测（在 ~/project-107-v2wave123 里跑，含对拍校验）
#    KD_TEACHER_CKPT 指向 v2wave123 的 best midpoint（如 outputs/midpoint_ep150.pt）
KD_PREDS_ONLY=1 KD_TEACHER_CKPT=outputs/midpoint_ep150.pt KD_TEACHER_DIR=outputs \
  ~/venv/bin/python3 -u main.py
#    → outputs/kd_teacher_preds_{train,val,test}.npy + 对拍打印（regret≈0.12% 即对齐正确）

# ② student 侧：跑可行性实验（4 槽并行 2×2，方法×seed）
bash setup_exp.sh v2kd123reg42      # KD_TEACHER_DIR 默认指向 ~/project-107-v2wave123/outputs
bash setup_exp.sh v2kd123rr42
bash setup_exp.sh v2kd123reg123
bash setup_exp.sh v2kd123rr123
#    （可加 KD_LAMBDA= KD_RANK_W= 环境变量覆盖默认 1.0/1.0）

# ③ 判读 Phase A 结果（midpoint regret），决定 Phase B/C
```
