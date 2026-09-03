# GNN 延迟排序项目（NetlistOpt 集成）——文档导航 / 新手入口

> **给接手人**：本文档是 docs/ 的**入口**。先读本文 → `GNN_PROJECT_REQUIREMENTS.md`(强制规则) → `OPERATIONS.md`(怎么跑/现状) → 再按需深入各专题。
> 最后更新：2026-09-03（16.11.32）。

## 1. 项目是什么（30 秒版）

用 **GNN 从电路网表预测 SPICE 级传播延迟**（avg_delay），为 **Rust 贪心优化器 NetlistOpt** 的候选做**排序/粗筛**：NetlistOpt 每个贪心窗口会枚举多个功能等价的候选实现（10-15 个变体），GNN 负责把「最快实现」排到前面，SPICE 只精排 top-K（省 ~90% 仿真）。

- **数据侧**：TransiLog 生成等价变体电路（含 SPICE 仿真结果），喂 GNN 训练；新一批 60 万行数据（V3）规格在 `DATA_SPEC_V2.md`。
- **模型侧**：6 层 GraphConv GNN + corner 注意力（cornerattn），STRUCT_MODE=logic_only（固定 10 类逻辑，杜绝 cell 名 OOV）。
- **Rust 侧**：NetlistOpt（独立仓库，仅本地）`tl_opt` 贪心 + shadow 并行评估（GNN vs SPICE 对照，46 电路基准）。
- **当前判定（2026-08-26 定）**：GNN = **启发式预排序**（严格 recall@3 47.2% 未达 90% 线），两阶段遗憾 2.02% 可接受。主判据见 `GNN_RUST_DATA_DIFF.md` §10。

## 2. 文档地图（先读顺序）

| 顺序 | 文档 | 管什么 | 何时读 |
|---|---|---|---|
| 1 | `README.md`（本文件） | 导航 / 概述 | 永远第一份 |
| 2 | `GNN_PROJECT_REQUIREMENTS.md` | **最高优先级要求**：版本号规范、触发文本、长期规则、行为约束、SSH/启动规范 | 任何新对话/动手前必读（§1-§5） |
| 3 | `OPERATIONS.md` | 运行手册：服务器、变体字典、开关表、当前状态、serve/shadow/git | 要跑训练/serve/shadow/分析前 |
| 4 | `OPEN_ISSUES.md` | 未决/待复核/风险集中清单 | 下结论/接手新任务前扫一眼 |
| 5 | `PROJECT_LOG.md` | 实验流水（含 16.x 记录区）+ Project Overview | 查某个实验的历史结果 |
| 6 | `GNN_RUST_DATA_DIFF.md` | 训练数据 vs Rust 候选差异的**实测与归因全集**（§10-§14）；数据相关问题以此为准 | 回答数据问题/查归因（强制，见 REQUIREMENTS §4） |
| 7 | `GNN_CODING_LESSONS.md` | 编码/部署踩坑实录 | 改数据加载/缓存/训练管线、部署服务器前 |
| 8 | `DATA_SPEC_V2.md` | 数据规格 V3（给生成方 + 接收验收） | 涉及数据生成/验收 |
| 辅助 | `DATA_SPEC.md`(V1 旧)、`DISTILL_PLAN.md`、`TESTING_GUIDE.md`、`V2_DATA_FULL_DELIVERY.md`、`DATA_SPEC_V2_ISSUES.md`、`STRUCTURAL_PATTERNS.md(_CN)`、`GIT_PUSH.md` | 各自专题/历史 | 需要时 |

> ⚠ 文档有大量「版本记录式」更新与互相引用；**同一结论若有新旧版本，以标注最新日期/版本者为准**；标「⚠/待复核」的不作定论（见 OPEN_ISSUES）。

## 3. 代码结构（根目录）

```
config.py            全局配置（模型/数据/开关，可被 env 覆盖）
main.py              -> src.train_sweep.main（训练/评估/KD 导出入口）
setup_exp.sh         服务器实验启动器（clone GitHub 分支 → sed 变体 → nohup 训练）
src/
  graph_builder.py    网表解析 → 图（parse_netlist、STRUCT_MODE 门特征、缓存构建）
  data_loader.py      parquet → 数据集（JSON pin 列、近似 ids_avg、group_ids/grp/row_idx）
  model.py            DelayGNN（6 层 GNN + corner 注意力 + 多输出读出）
  train_sweep.py      训练循环（huber + 可选 RANK_LOSS_W 排序 loss + KD 蒸馏 loss）
  utils.py            切分(split_by_expr)、排序指标、GroupedBatchSampler
data/                训练数据（batch_v2_full/rest/m4/io、delivery*、sc_expansion.json 等）
scripts/diag/        分析/诊断脚本（_ 前缀；shadow 分析、serve、数据核对等）
outputs/ models/ reports/   模型/报告
NetlistOpt/          独立 git 仓库（Rust，不推远端，见下）
```

## 4. 两个 git 仓库

| 仓库 | 位置 | 远端 | 规则 |
|---|---|---|---|
| project（本仓库） | 根目录 | `github.com/huifeng382/-project.git` 分支 `10.3.3-fix-earlystop` | 本地 + GitHub；版本号提交规范见 REQUIREMENTS §1 |
| NetlistOpt | `NetlistOpt/` | **无** | 仅本地 git，不推远端（用户明确要求） |

## 5. 新手路径建议

1. 读本 README → REQUIREMENTS §1-§5（版本号/规则/行为约束）；
2. 看 `OPERATIONS.md`「当前状态」节：服务器上正在跑什么、哪些是基线；
3. 读 `GNN_RUST_DATA_DIFF.md` §10（Rust 判定）→ §12/§13（数据差异实测）→ §14（审计，含 ⚠ 项）；
4. 想跑实验：OPERATIONS 变体字典 + setup_exp.sh；想写代码：CODING_LESSONS + src/ 对应文件。
