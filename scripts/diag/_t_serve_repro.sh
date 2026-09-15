#!/bin/bash
# _t_serve_repro.sh — serve 进程间可复现性验收（17.2.4 根因修复的判决实验）。
#
# 做什么：同一个 ckpt、两个**全新 serve 进程**、**刻意不钉任何 env**，各跑一趟 level2 分片，
#         然后逐字节比对产出的 gnn_shadow.csv（md5 清单，文件名 + md5 都参与比对）。
# 判据：两趟清单完全相同 → 通过（= 边序不再随 PYTHONHASHSEED 变，源头已定序）。
#       有差异 → 通道没堵住；此时差异应仍是平均秩 ±0.5 的整跳，把打印的差异行贴回来。
#
# 为什么这一条就够定因：本版本只改了 parse_netlist 的边序构造
# （list(set(edges)) → sorted(set(edges))），ckpt / scaler / Rust / 模拟器 / 缓存全不变。
# 两趟唯一的差别是 serve 进程身份；**不钉 env 还能逐位一致**，就只能是边序不再依赖哈希种子。
#
# 用法（服务器，一条命令；参数：ckpt scaler [level]）：
#   bash ~/-project/scripts/diag/_t_serve_repro.sh ~/project-107-v2nowave42m4/outputs/midpoint_ep250.pt ~/project-107-v2nowave42m4/outputs/scaler.pkl
# 成本：约 3 分钟（两次 serve 启动 + 两趟 level2 分片；候选集与真值列全走 XYCE_CACHE，几乎不耗 Xyce）。
# 注意：脚本结束时留下最后一趟的 serve 在 8000 —— 若不是交付基线，按 OPERATIONS §6.7 换回。
# 注意：不会碰 ~/NetlistOpt/temp_sim_cache/（内容寻址延迟缓存），只清 temp_sim_test/tl_opt_batch。

set -u
CKPT="${1:?用法: $0 CKPT SCALER [LEVEL]}"
SCALER="${2:?缺少 SCALER}"
LEVEL="${3:-level2}"
NL="$HOME/NetlistOpt"
BATCH="$NL/temp_sim_test/tl_opt_batch"

[ -f "$CKPT" ]   || { echo "ERROR: 找不到 $CKPT"; exit 1; }
[ -f "$SCALER" ] || { echo "ERROR: 找不到 $SCALER"; exit 1; }
[ -d "$NL" ]     || { echo "ERROR: 找不到 $NL"; exit 1; }

run_once() {   # $1 = 轮次标签 A/B
  local TAG="$1" n=0 tot=0 f
  # ---- 1) 全新 serve 进程（不钉任何 env —— 这是本探针的要害）----
  pkill -f 'serve_htt[p].py' 2>/dev/null
  for i in $(seq 15); do ss -ltn 2>/dev/null | grep -q ':8000' || break; sleep 1; done
  ( cd "$HOME/-project" && env -u USE_IDS_AVG_APPROX -u IDSGNN_CKPT -u GNN_PORT2 \
      nohup "$HOME/venv/bin/python3" scripts/diag/serve_http.py \
      --ckpt "$CKPT" --scaler "$SCALER" --port 8000 \
      > "$HOME/repro_serve_${TAG}.log" 2>&1 & )
  for i in $(seq 40); do ss -ltn 2>/dev/null | grep -q ':8000' && break; sleep 1; done
  if ! ss -ltn 2>/dev/null | grep -q ':8000'; then
    echo "ERROR: [${TAG}] serve 未就绪（看 ~/repro_serve_${TAG}.log）"; exit 1
  fi
  echo "  [${TAG}] serve: $(pgrep -af 'serve_htt[p].py' | head -1)"
  echo "  [${TAG}] PYTHONHASHSEED=${PYTHONHASHSEED:-未设}  OMP_NUM_THREADS=${OMP_NUM_THREADS:-未设}  MKL_NUM_THREADS=${MKL_NUM_THREADS:-未设}"

  # ---- 2) 跑一趟 level2 分片（前台，跑完才往下走 → 天然没有"旧文件"坑）----
  rm -rf "$BATCH"          # 必须清：CSV 是 append 模式，旧行会混进本趟
  ( cd "$NL" && TL_ONLY="$LEVEL" GNN_SHADOW=1 GNN_HOST=127.0.0.1 GNN_PORT=8000 SPICEVIZ_OFF=1 \
      timeout 900 cargo test --release --test tl_opt_shadow_batch -- --nocapture --ignored \
      > "$HOME/repro_shadow_${TAG}.log" 2>&1 )
  if ! grep -q 'test result: ok' "$HOME/repro_shadow_${TAG}.log"; then
    echo "ERROR: [${TAG}] cargo 未报 ok，看 ~/repro_shadow_${TAG}.log 末尾（尾部 5 行）"
    tail -5 "$HOME/repro_shadow_${TAG}.log" | sed 's/^/    /'
    exit 1
  fi

  # ---- 3) 收 md5 清单（文件名参与比对 → 顺带验产物集合一致）----
  : > "$HOME/repro_${TAG}.md5"
  while IFS= read -r f; do
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "${f#$BATCH/}" >> "$HOME/repro_${TAG}.md5"
    n=$((n + 1))
    tot=$((tot + $(wc -l < "$f")))
  done < <(find "$BATCH/$LEVEL" -name gnn_shadow.csv 2>/dev/null | sort)
  echo "  [${TAG}] CSV 数=$n  总行数=$tot"
  [ "$n" -gt 0 ] || { echo "ERROR: [${TAG}] 一个 CSV 都没产出（$BATCH/$LEVEL）"; exit 1; }
}

echo "############ serve 可复现性验收：LEVEL=$LEVEL，两个全新进程，不钉 env ############"
run_once A
run_once B

echo
echo "== 两趟 CSV md5 清单比对（$LEVEL）=="
if diff -u "$HOME/repro_A.md5" "$HOME/repro_B.md5" > "$HOME/repro_diff.txt"; then
  echo "结果: 逐字节相同 ✅ —— 通道已堵住（不钉 PYTHONHASHSEED 也一致）"
else
  echo "结果: 有差异 ❌ —— 通道未堵住，差异清单（前 40 行）:"
  sed -n '1,40p' "$HOME/repro_diff.txt" | sed 's/^/    /'
  echo "  （完整差异: ~/repro_diff.txt；下一步应看首个差异 CSV 的 gnn_pred 列是否仍为 ±0.5 整跳）"
fi
echo "清单: ~/repro_A.md5 / ~/repro_B.md5   日志: ~/repro_shadow_A.log / ~/repro_shadow_B.log"
echo "末尾 serve 挂的是: $CKPT（若非交付基线，按 OPERATIONS §6.7 换回 42m4 ep250）"
