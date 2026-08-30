# V3 full+rest 数据 wave 消融报告（16.8.1，2026-08-29）

## 基本信息

- **数据**：`batch_v2_full` + `batch_v2_rest`（默认 full+rest；`batch_v2_io` 作 dev/val）
- **代码**：16.4.0~16.8.0（内存修复 / RESUME / master 缓存），`STRUCT_MODE=logic_only`
- **切分**：`SPLIT_SEED=42`（按 expr 分组）；`TRAIN_SEED=42`
- **训练**：4 run 并行（auto_launch + master 缓存），**统一数据 mtime / 统一离群点掩码 / 统一代码**
- **离群点掩码**：同一份 `outlier_keep_5dd789ff66a3.npy`（wave42 基训产出，其余 3 run 命中共享）

## 4 个变体

| 变体 | 配置 |
|---|---|
| `v2wave42` | wave 开，3 字段 `ids_avg,ids_peak,vds_swing`，覆盖率 1.0 |
| `v2nowave42` | wave 关（Rust 部署基线） |
| `v2ia42` | wave 开，**单字段 `ids_avg`** |
| `v2cov2542` | wave 开，3 字段，**行覆盖率 0.25**（WAVE_COVERAGE_SEED=42） |

## 结果（Test 评估；checkpoint 按 val smoothed_rel_err 选点）

| 指标 | wave（3字段） | nowave | **ids_avg 单字段** | 25% 覆盖率 |
|---|---|---|---|---|
| Test Median Rel Err | 11.89% | 23.98% | **9.02%** | 29.11% |
| 选择遗憾（全局） | 0.65% | 3.77% | **0.55%** | 32.90% |
| recall@3 全局 A/B | 84.6 / 93.7 | 62.7 / 81.0 | **87.1 / 96.6** | 40.3 / 73.8 |
| recall@3 spread>10% A/B | 93.5 / 97.9 | 73.6 / 88.1 | **96.7 / 99.4** | 46.9 / 78.9 |
| Spearman（全局） | 0.719 | 0.386 | **0.745** | 0.217 |
| V2-full(4pin) | 13.4% | 39.6% | **12.1%** | 39.4% |
| 停止 epoch | 54 | 152 | 236 | 162 |

（A = 严格：真实 #1 ∈ 预测 top-K；B = 宽松：预测 top-K ∩ 真实 top-K 非空）

## 结论

1. **ids_avg 单字段最优**（遗憾 0.55%、recall@3 B 96.6%、Test Median 9.02%）——比全 wave 略好，省 2/3 wave 字段
2. **全 wave 次之**（0.65% 遗憾）——与 ia42 同一梯队
3. **nowave 明显差**（3.77% 遗憾，约 7 倍）——wave 特征贡献巨大
4. **25% 覆盖率有害**（32.9% 遗憾，灾难）——与旧数据消融结论一致，坐实「省仿真量不可行」

## Rust 集成决策（仿真预算原则）

> **原则（用户定）**：GNN 的价值 = 节省仿真预算。Rust 部署若还要跑完整波形仿真就失去意义 → **wave 数据只作参考/上限，不用于真实 Rust 部署**。

| 方案 | Rust 侧成本 | 遗憾（full+rest） | 定位 |
|---|---|---|---|
| nowave | 0（不跑波形仿真） | 3.77% | **Rust 部署基线** |
| 真实 ids_avg | 需波形仿真（与 GNN 目的冲突） | 0.55% | 上限参照 |
| **ids_avg 近似公式** | 近零（几个 FLOP） | 待验证 | 候选升级 |
| 25% 覆盖率 | 省 75% 仿真量 | 32.9% | 不可行（已证伪） |

### ids_avg 近似公式（候选，待验证）

物理：`ids_avg ≈ C_L × ΔV / T_sw`（平均漏极电流 = 输出负载充放电电荷 ÷ 开关时间）

| 量 | 便宜代理（Rust 端网表/静态量可得） |
|---|---|
| C_L | 扇出 × 输入电容 + `parasitic_caps_json` + 连线估 |
| ΔV | `VDD × (1 − exp(−T_sw/(R_on·C_L)))`，R_on 从驱动强度/晶体管尺寸估（捕获 vds_swing 物理） |
| T_sw | 输入 slew + 输出 RC（≈ slew + R_on·C_L） |

**边界**：近似只能恢复「便宜量与真实 ids_avg 相关部分」；晶体管级效应（短路电流/速度饱和/体效应）拿不到 → 期望部分恢复（3.77% → 2-3%？），到不了 0.55%。若近似只用 GNN 已有特征则几乎不加信息（GNN 可自学习该映射）——必须编码新派生量（每门负载×斜率×摆幅、RC 时间常数）。

**验证实验**：① 用数据内真实 ids_avg 拟合近似公式（评估 R²）；② 以近似 ids_avg 作特征训一版测遗憾。遗憾 ≈0.6-1.5% → Rust 端实现；≈3.5% → 不值，用 nowave。

## 模型文件（服务器）

- `v2ia42`：`~/project-107-v2ia42/outputs/best_model.pt`（候选交付，需 Rust 能提供 ids_avg）
- `v2nowave42`：`~/project-107-v2nowave42/outputs/best_model.pt`（Rust 基线）
- scaler：各自 `outputs/scaler.pkl`；serve 端 `STRUCT_MODE=logic_only`

## Rust 验证进度

- [ ] nowave 46 候选 recall@3（严格/宽松 k=2/3 + 两阶段遗憾）——进行中
- [ ] ids_avg 近似公式拟合 + 训练实测
