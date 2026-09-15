#!/bin/bash
# _shadow_ckpt_sweep.sh — 扫一个 nowave 臂已存的多个 ckpt：逐个换 serve 跑 shadow，按 epoch 存盘。
#
# 目的：检验「服务 Best midpoint（按 val loss 挑）」是否等于部署最优。
#   2026-09-15 发现 42b(ep100) 与 42m4(ep250) 训练侧几乎同级（遗憾 3.94% vs 3.80%）
#   但部署口径差 3.60pp（14.26% vs 10.66%），头号嫌疑就是 ckpt 选点依据是 val loss。
#   本脚本把该臂每个 ckpt 的部署口径指标都测出来 → 看 epoch-部署质量曲线的形状。
# 成本：每趟约 2min（候选集与真值列全走 XYCE_CACHE，几乎不耗 Xyce）。
#
# ⚠ 分辨率（2026-09-15 实测并定因；此前给过的解释全部作废，勿再引用）：
#   同一 ckpt 连测多次 = 10.51 / 10.60 / 10.66 / 10.86 / 11.13% —— 跨度 0.62pp。
#   根因 = **边序随哈希种子变**：src/graph_builder.py 的 parse_netlist 原为 list(set(edges))，
#   字符串元组集合的迭代序随 PYTHONHASHSEED 变 → edge_index 行序变 → 前向的 float32 累加序变
#   → 同一候选跨进程预测差 float32 的 1 ulp 量级（~1e-7 相对）→ 近并列候选（大量候选本就是
#   同一份网表，float32 下位级相等）互换名次 → 平均秩 (i+j)/2+1 整组 ±0.5 → top-1 翻转 →
#   该集 0.00% ↔ 19.92%。
#   定因链（都是双向对照，不是相关）：同一个 serve 进程连跑两趟 CSV 逐字节一致；**只换进程**
#   就有 239/4190 行变化、全是 ±0.5 整跳；钉 PYTHONHASHSEED 后两个新进程逐字节一致；
#   三个线程变量（OMP/MKL_NUM_THREADS/MKL_DYNAMIC）钉与不钉**零差别** → 与线程无关。
#   已修（17.2.4）：边序规范化 sorted(set(edges))（源头定序）+ 启动行加 PYTHONHASHSEED=0 兜底；
#   同时**撤销** 17.2.3 的相对容差 —— 容差取 1e-9，比真实抖动小两个量级，实测无效。
#   **作废的旧解释**：(a)「gnn_pred 只存 7 位有效数字」——它是平均秩，值就是 2.5/3.0 这种
#   半整数，7 位一位没丢；(b)「噪声底 0.3pp / 上限 0.38pp」——实测跨度 0.62pp；
#   (c)「17.2.3 的相对容差已修」——1e-9 瞄错了量级。
#   所以旧数据：<1pp 当打平，1.5pp 以上才是真差。17.2.4 之后每个 ckpt 一次即可。
#
# 用法：bash ~/-project/scripts/diag/_shadow_ckpt_sweep.sh ARM_DIR ARM_TAG [EPOCHS...]
#   bash ~/-project/scripts/diag/_shadow_ckpt_sweep.sh ~/project-107-v2nowave42b v2nowave42b 50 100 150 200 250
#   省略 EPOCHS 则默认 50 100 150 200 250
# 输出：~/sweep_<ARM_TAG>_ep<N>.out（每个 ckpt 一份完整聚合结果）+ ~/serve_<ARM_TAG>_ep<N>.log
# 注意：脚本结束时会留下最后一趟的 serve 在 8000 —— 若不是交付基线，记得按 OPERATIONS §6.7 换回。

set -u
ARM_DIR="${1:?用法: $0 ARM_DIR ARM_TAG [EPOCHS...]}"
ARM_TAG="${2:?缺少 ARM_TAG}"
shift 2
EPOCHS=("$@")
[ "${#EPOCHS[@]}" -gt 0 ] || EPOCHS=(50 100 150 200 250)

SCALER="$ARM_DIR/outputs/scaler.pkl"
[ -f "$SCALER" ] || { echo "ERROR: 找不到 $SCALER"; exit 1; }
[ -d "$HOME/NetlistOpt" ] || { echo "ERROR: 找不到 ~/NetlistOpt"; exit 1; }

echo "############ 扫描 $ARM_TAG ：epochs ${EPOCHS[*]} ############"

for CK in "${EPOCHS[@]}"; do
  CP="$ARM_DIR/outputs/midpoint_ep${CK}.pt"
  OUT="$HOME/sweep_${ARM_TAG}_ep${CK}.out"
  if [ ! -f "$CP" ]; then echo "=== ep${CK}: 缺 ckpt，跳过 ==="; continue; fi

  # ---- 1) 换 serve：杀掉所有旧 serve，等端口释放，再起新的 ----
  # env -u …：显式清掉 ids/第二端点相关变量。42b/42m4 都是纯拓扑臂，父 shell 里若残留
  # USE_IDS_AVG_APPROX 或 IDSGNN_CKPT，serve 就不再是对照臂了（此前用 /proc/PID/environ 验证过这点）。
  # PYTHONHASHSEED=0：17.2.4 兜底（根因已由 sorted(set(edges)) 在源头定序，这里防其它哈希序依赖）。
  pkill -f 'serve_htt[p].py' 2>/dev/null
  for i in $(seq 15); do ss -ltn 2>/dev/null | grep -q ':8000' || break; sleep 1; done
  ( cd "$HOME/-project" && env -u USE_IDS_AVG_APPROX -u IDSGNN_CKPT -u GNN_PORT2 PYTHONHASHSEED=0 \
      nohup "$HOME/venv/bin/python3" scripts/diag/serve_http.py \
      --ckpt "$CP" --scaler "$SCALER" --port 8000 \
      > "$HOME/serve_${ARM_TAG}_ep${CK}.log" 2>&1 & )
  for i in $(seq 40); do ss -ltn 2>/dev/null | grep -q ':8000' && break; sleep 1; done
  if ! ss -ltn 2>/dev/null | grep -q ':8000'; then
    echo "=== ep${CK}: serve 未就绪，跳过（看 ~/serve_${ARM_TAG}_ep${CK}.log）==="
    continue
  fi

  # ---- 2) 跑 shadow ----
  # 删旧结果文件：收尾 waiter 是整块 { … } > ~/shadow_analyze.out（结尾才写），
  # 所以「文件重新出现且含分析段」才是本趟完成的可靠信号。
  # ⚠ 不要用「分片归零 + sleep N」判收尾：waiter 每 30s 才轮询一次、再 sleep 5，
  #   分片归零到落盘之间有最长 ~35s 黑洞，那时读到的是上一趟的文件（OPERATIONS §6.6 的坑）。
  rm -f "$HOME/shadow_analyze.out"
  unset GNN_PORT2 USE_IDS_AVG_APPROX IDSGNN_CKPT
  if ! bash "$HOME/-project/scripts/diag/run_shadow_batch.sh"; then
    echo "=== ep${CK}: run_shadow_batch 启动失败，跳过 ==="
    continue
  fi
  sleep 20   # 让分片真正起进程；run_shadow_batch 是后台起分片后立刻返回的
  for i in $(seq 300); do pgrep -f 'car[g]o test --release --test tl_opt_shadow_batch' >/dev/null 2>&1 || break; sleep 10; done
  for i in $(seq 180); do grep -q '选择遗憾（' "$HOME/shadow_analyze.out" 2>/dev/null && break; sleep 5; done
  if ! grep -q '选择遗憾（' "$HOME/shadow_analyze.out" 2>/dev/null; then
    echo "=== ep${CK}: 等超时，结果文件未落盘 —— 检查 ~/shadow_lvl*.log ==="
    continue
  fi

  cp "$HOME/shadow_analyze.out" "$OUT"
  echo "=== ep${CK} 完成 → $OUT ==="
  # 笼统匹配 `前k名中出现实际` 一次覆盖严格与宽松两行（宽松是必需项，别只给严格）
  grep -E '候选集数|前k名中出现实际|选择遗憾（|两阶段最终遗憾（|Spearman:' "$OUT" | sed 's/^/    /'
done

# ---- 3) 汇总：epoch-部署质量曲线 ----
# 严格与宽松两个 recall 口径都是必需项（只给严格算漏），表头用 ASCII 以免多字节对不齐。
echo
echo "############ $ARM_TAG 汇总（选择遗憾，越低越好；<0.3pp 视为打平）############"
pick() { grep -m1 "$1" "$f" | sed "$2"; }
printf '%-7s %-9s %-9s %-9s %-9s %-9s %s\n' ckpt regret strict_k2 strict_k3 loose_k2 loose_k3 spearman
for CK in "${EPOCHS[@]}"; do
  f="$HOME/sweep_${ARM_TAG}_ep${CK}.out"
  if [ ! -f "$f" ]; then printf 'ep%-5s 未产出\n' "$CK"; continue; fi
  R=$(pick '选择遗憾（' 's/.*: *//; s/ *(达标.*//')
  S=$(pick 'Spearman:' 's/.*Spearman: *//; s/ *(次判据.*//')
  S2=$(pick '前k名中出现实际第1名' 's/.*k=2 *//; s/ *k=3.*//')
  S3=$(pick '前k名中出现实际第1名' 's/.*k=3 *//')
  L2=$(pick '前k名中出现实际前k之一' 's/.*k=2 *//; s/ *k=3.*//')
  L3=$(pick '前k名中出现实际前k之一' 's/.*k=3 *//')
  printf 'ep%-5s %-9s %-9s %-9s %-9s %-9s %s\n' "$CK" "$R" "$S2" "$S3" "$L2" "$L3" "$S"
done
echo "############ $ARM_TAG 扫完 ############"
