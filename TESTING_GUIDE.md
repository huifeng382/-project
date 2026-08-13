# 测试流程指南

## 1. 环境准备

**服务器**：tianlang@orca（24 核 / 60GB RAM / Python 3.13 / venv 在 `~/venv`）。

**本地**：仅做代码编辑、只读分析和 `git push`。训练只在服务器上跑。

---

## 2. 启动测试

所有实验通过 `setup_exp.sh` 一键启动（clone → 改配置 → 后台训练）。

### 基本命令
```bash
pkill -9 -f main.py                      # 清理旧训练进程
pkill -9 -f _resume.py                   # 清理续跑进程
cd ~/exp107 && git pull                 # 拉取最新代码和脚本
bash setup_exp.sh <变体名>              # 启动一个实验（自动 clone 到 ~/project-107-<变体>）
```

### 常用变体

| 变体 | 说明 |
|---|---|
| `rank` | 当前最优基线（cornerattn + wave + struct_prior，默认配置，TRAIN_SEED=42） |
| `seed123` / `seed2024` / `seed456` / `seed789` / `seed1357` / `seed2468` / `seed3579` / `seed9012` | 不同 TRAIN_SEED（集成/裁剪平均用） |
| `anneal` | 更深退火（LR_MIN 1e-7, LR_FACTOR 0.4） |
| `waverich` | 丰富晶体管波形聚合（mean+max+std） |
| `cornerattn` | Corner 注意力池化 |
| `rankloss1` | 成对排序损失 w=0.5 |
| `newcaps` / `newwave` / `newnoise` | 新物理特征消融（寄生电容 / 晶体管波形 / 电源噪声） |

> **注意**：`rank` 变体用默认 TRAIN_SEED=42，是集成的基础之一。其他 seed 变体只改 TRAIN_SEED，其余配置与 rank 完全一致，可直接集成。

### 中途快照（midpoint）
- `SAVE_MIDPOINTS=True`（默认开启）→ 每 50 epoch 存一个 `midpoint_ep{N}.pt`。
- 训练结束后，代码自动在 SUMMARY 前评估所有 midpoint，按加权得分选最优 epoch 作为最终模型。
- 不影响训练 RNG（midpoint 代码在训练循环结束后、SUMMARY 前，用局部 import）。

---

## 3. 检查运行状态

### 确认四个槽都在跑 + 同一切分
```bash
for d in <var1> <var2> <var3> <var4>; do
  L=~/project-107-$d/train107$d.log
  echo "== 107-$d =="; grep "划分\|in_dim\|x shape" "$L" 2>/dev/null | head -2
  tail -1 "$L"
done
```

### 查看当前 epoch 进度
```bash
for d in <var1> <var2> <var3> <var4>; do
  echo "== 107-$d =="; tail -2 ~/project-107-$d/train107$d.log
done
```

### 查看服务器 CPU 占用
```bash
ps -eo pid,user,%cpu,args --sort=-%cpu | grep "main.py" | head -12
```

---

## 4. 查看结果

### 单个跑完的 SUMMARY
```bash
tail -60 ~/project-107-<var>/train107<var>.log
```

### 批量 SUMMARY（四个都跑完后）
```bash
for d in <var1> <var2> <var3> <var4>; do
  L=~/project-107-$d/train107$d.log
  echo "========== 107-$d =========="
  grep -q "^SUMMARY" "$L" && sed -n '/^SUMMARY/,$p' "$L" || echo "  未完成"
  echo
done
```

---

## 5. 结果解读

### SUMMARY 中最重要的指标（按优先级）

**第一优先——排序（真实任务：等价变体择优）**
```
[排序 spread>10%] 组(>=2)=N  Spearman=X(→1)  选择遗憾=X%(→0)  top1=X%(→100)  捕获率=X%(→100)
```
- **选择遗憾**：核心 KPI——模型选中的比最优慢多少。越低越好，0% = 每次都挑对。
- **Spearman**：组内排序一致性，越高越好，1 = 完美。
- **top1 命中**：是否挑中精确最优，越高越好。

**第二优先——成对分辨**
```
[成对分辨(按真实延迟差)] <2%:X%(n)  2-5%:X%(n)  5-10%:X%(n)  >10%:X%(n)
```
- `>10%` 档反映模型对大差异的可靠分辨能力。`<2%` 档受 SNR 天花板限制，~50% 随机。

**第三优先——点精度**
```
Test Median Rel Err: X%(→0)   Mean Abs Err: Xps(→0)
```
- 逐样本预测精度，辅助参考。Mean Rel Err 被小延迟放大，仅参考。

### 配置确认
```
Config: ... BEST_METRIC=... SPLIT_SEED=... TRAIN_SEED=...
```
- `TRAIN_SEED` 不同 → 做集成用。
- `SPLIT_SEED=42` 固定 → 同切分可比。

---

## 6. 集成评估

### 4-seed 等权平均（验证 13.6 现成结果）
```bash
cd ~/project-107-rank && git pull && ~/venv/bin/python3 _ens4.py
```
读 rank/seed123/seed2024/seed456 的 test_predictions.npz，输出 4-seed 平均 + 各单 seed 指标。

### 裁剪平均（当前最优，去掉遗憾最差的 seed）
```bash
cd ~/project-107-rank && git pull && ~/venv/bin/python3 _trim.py
```
自动按遗憾排序，对比「全量平均 / Trim 1（去最差）/ Trim 2（去最差两个）」三种。当前最优是 Trim 1（去 rank，留 456+2024+123）= 遗憾 2.08%。

### 集成要点
- 同配置不同 TRAIN_SEED 才能集成（SPLIT_SEED 固定 42 → 同一切分可比）。
- 误差相关时，裁剪平均（去最差 seed）优于全量平均。
- 脚本读 `~/project-107-<变体>/outputs/test_predictions.npz`，需先确认这些 npz 存在。

---

## 7. 清理旧实验目录

已跑完的实验，如果结果已记录在 PROJECT_LOG.md，可用以下命令清理：
```bash
# 确认 log 目录存在（防止误操作）
ls ~/project-107-<var>/train107<var>.log
# 安全删除
rm -rf ~/project-107-<var>
```

日志和预测文件会丢失，建议先 `tail -60` 确认结果已记录再删。

---

## 8. 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `RuntimeError: received 0 items of ancdata` | 文件描述符不足 | 已在 setup_exp 中加 `ulimit -n 8192` |
| `ModuleNotFoundError: No module named 'torch'` | 后台进程没用到 venv | setup_exp 已改用 `~/venv/bin/python3` |
| Train Loss = nan | 成对排序损失崩 | 检查 RANK_LOSS_W 是否过大 |
| 训练卡住不动 | DataLoader 加载慢 | 调高 `num_workers`（见 #12） |
| epoch 耗时异常 | 4 槽并行 CPU 竞争 | 用 `ps -eo` 检查是否有残留进程或他人占用 |
