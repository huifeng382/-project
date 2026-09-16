#!/bin/bash
# run_shadow_batch.sh — Rust 46 候选 shadow 基准标准启动（16.9.0，防两个坑）
#
# 坑1（慢）：批次默认串行 + Xyce 单核 → 24 核只用一个核，46 电路排队数小时。
#   对策：按 level 分片（level0-3 各一进程）+ level4 按电路逐个进程，全并行。
# 坑2（错）：gnn_shadow.csv 是 append 模式且路径固定（temp_sim_test/tl_opt_batch），
#   旧 run 的行会混入 → 分析结果错误（曾混入 8/27 旧模型 ~1100 行）。
#   对策：启动前**归档**该目录（必须在分片启动前做一次，不能每个分片各清一次）。
#   17.3.16：原为 rm -rf —— 等于每跑一趟就毁掉上一趟的原始数据。后果是「五点重扫」
#   变成「第五趟删掉第四趟」，且历史各 ckpt 的配对分析永久不可得（2026-09-16 只能靠
#   ~/sweep_*.out 里残留的 106 行明细倒推）。改为 mv 到 ~/shadow_archive/<TAG>_<时间戳>/。
#   TAG 默认取当前 serve 的 ckpt 名（--ckpt 的 basename 去 .pt），可用 SHADOW_TAG 覆盖。
#   归档目录保留 <root>/<level>/<stem>/gnn_shadow.csv 原布局 → **本身就是合法 --root**，
#   可事后直接 `_shadow_analyze.py --root ~/shadow_archive/<TAG>_<时间戳>`。
#   存档不自动清理，占盘自己看情况删（CSV 很小，单趟 MB 级）。
#
# 前置：GNN serve 已运行（
#   nohup ~/venv/bin/python3 ~/-project/scripts/diag/serve_http.py \
#     --ckpt <model.pt> --scaler <scaler.pkl> --port 8000 & ）
# 用法：bash ~/-project/scripts/diag/run_shadow_batch.sh
# 结果：全部跑完自动执行 _shadow_analyze.py，输出到 ~/shadow_analyze.out

set -u
NL="$HOME/NetlistOpt"
[ -d "$NL" ] || { echo "ERROR: 没有 $NL"; exit 1; }

# 0) 检查 GNN serve（括号技巧防自匹配）
if ! pgrep -f 'serve_htt[p]' >/dev/null; then
  echo "ERROR: GNN serve 未运行。先启动："
  echo "  nohup ~/venv/bin/python3 ~/-project/scripts/diag/serve_http.py --ckpt <model.pt> --scaler <scaler.pkl> --port 8000 &"
  exit 1
fi

cd "$NL"

# 1) 归档旧 CSV（防污染——append 模式，旧行混入会让分析结果错；但不再销毁）
ARCHIVE_ROOT="$HOME/shadow_archive"
if [ -e temp_sim_test/tl_opt_batch ]; then
  # TAG：优先 SHADOW_TAG，否则取 serve 的 --ckpt basename（去掉 .pt），再否则 run
  SERVE_ARGS=$(ps -o args= -p "$(pgrep -f 'serve_htt[p]' | head -1)" 2>/dev/null | head -1 || true)
  CKPT=$(printf '%s\n' "${SERVE_ARGS:-}" | sed -n 's/.*--ckpt[= ][ ]*\([^ ]*\).*/\1/p' | head -1)
  TAG=$(basename "${CKPT:-run}" .pt); [ -n "$TAG" ] || TAG=run
  [ -n "${SHADOW_TAG:-}" ] && TAG="$SHADOW_TAG"
  TS=$(date +%Y%m%d_%H%M%S)
  DEST="$ARCHIVE_ROOT/${TAG}_$TS"
  mkdir -p "$ARCHIVE_ROOT"
  n=1
  while [ -e "$DEST" ]; do DEST="$ARCHIVE_ROOT/${TAG}_${TS}_$n"; n=$((n + 1)); done
  if ! mv temp_sim_test/tl_opt_batch "$DEST"; then
    echo "ERROR: 归档失败（$DEST）—— 拒绝对未清理的旧 CSV 继续跑（append 会混趟）"
    exit 1
  fi
  {
    echo "时间       : $(date +%F\ %T)"
    echo "TAG        : $TAG"
    echo "源目录     : $NL/temp_sim_test/tl_opt_batch"
    echo "serve 进程 : ${SERVE_ARGS:-未取到}"
    echo "serve ckpt : ${CKPT:-未识别}"
    echo "repo rev   : $(cd ~/-project 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo 未知)"
    echo "repo dirty : $(cd ~/-project 2>/dev/null && git status --porcelain 2>/dev/null | wc -l || echo '?') 个改动"
  } > "$DEST/RUN_INFO.txt"
  echo "[$(date +%F\ %T)] 已归档旧 CSV → $DEST（附 RUN_INFO.txt）"
else
  echo "[$(date +%F\ %T)] 无旧 CSV 目录，跳过归档（首跑）"
fi

# 2) 并行分片：level0-3 各一个进程；level4 按电路逐个进程（大电路最慢，全并行）
for l in 0 1 2 3; do
  TL_ONLY=level$l GNN_SHADOW=1 GNN_HOST=127.0.0.1 GNN_PORT=8000 SPICEVIZ_OFF=1 \
    nohup cargo test --release --test tl_opt_shadow_batch -- --nocapture --ignored \
    > ~/shadow_lvl$l.log 2>&1 &
done
for c in $(ls testbench/tl_cells/level4/*.tl | xargs -n1 basename | sed 's/\.tl$//'); do
  # 16.11.6: level4/ 前缀精确匹配（防 OVF 误带 ADD4_OVF/ovf1 → 并发写同 CSV 损坏）
  TL_ONLY=level4/$c GNN_SHADOW=1 GNN_HOST=127.0.0.1 GNN_PORT=8000 SPICEVIZ_OFF=1 \
    nohup cargo test --release --test tl_opt_shadow_batch -- --nocapture --ignored \
    > ~/shadow_lvl4_$c.log 2>&1 &
done
echo "[$(date +%F\ %T)] 已启动并行分片（level0-3 + level4 每电路一个进程）"

# 3) 自动收尾：所有 cargo 分片结束后跑分析
#    轮询用 cargo 模式（不要锚定二进制哈希——cargo 重编译后哈希会变）
nohup bash -c 'while pgrep -f "car[g]o test --release --test tl_opt_shadow_batch" >/dev/null 2>&1; do sleep 30; done; sleep 5; { echo "[$(date +%F\ %T)] 全部分片结束"; cd ~/-project && ~/venv/bin/python3 scripts/diag/_shadow_analyze.py --root ~/NetlistOpt/temp_sim_test/tl_opt_batch; } > ~/shadow_analyze.out 2>&1' > /dev/null 2>&1 &
echo "[$(date +%F\ %T)] 自动收尾已挂（完成后写 ~/shadow_analyze.out）"
echo "监控: tail -f ~/shadow_analyze.out"
