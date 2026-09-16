#!/bin/bash
# shadow_campaign.sh — 多 ckpt 串行 shadow 战役（无人值守）
#
# 目的：把「换 serve → 扫一趟 → 收分析输出」串成一条链，一夜跑完多个 ckpt，
#      每趟的数据都带正确来源（标签随数据走，见 run_shadow_batch.sh 17.3.17）。
#
# 每趟循环做五件事：
#   ① 重启 serve 钉到本 ckpt，用 GET / 核对 n_models==1（传多个 --ckpt 会变等权集成，
#      读数含义完全不同，必须挡住）
#   ② bash run_shadow_batch.sh —— 它会**归档上一棵树**（标签取自那棵树自己的 RUN_INFO，
#      即上一轮 ckpt，天然正确），并给本轮的树写入本轮 RUN_INFO
#   ③ 等分片与收尾守卫全部退出（守卫活到分析器跑完，所以这一等等于「等到分析都做完」）
#   ④ 校验：46 个 CSV、分析输出非空；不合格 → 整棵树隔离到 ~/shadow_archive_failed/
#   ⑤ 转存 ~/shadow_analyze.out → ~/sweep2_<tag>.out
#
# 安全阀（无人值守必需）：
#   · 磁盘剩余 < SHADOW_MINFREE_GB（默认 50G）→ 中止战役，不写满盘
#   · 单趟超过 SHADOW_MAXWAIT（默认 6h）→ 杀分片、隔离该树、继续下一个
#   · 失败/超时的树**永不进 ~/shadow_archive/**，避免半趟数据污染后续分析
#
# 用法（服务器）：
#   bash ~/-project/scripts/diag/shadow_campaign.sh                    # 自动发现 midpoint_ep*.pt
#   bash ~/-project/scripts/diag/shadow_campaign.sh --with-best        # 外加 best_model.pt
#   bash ~/-project/scripts/diag/shadow_campaign.sh --skip midpoint_ep250   # 已在跑的跳过
#   bash ~/-project/scripts/diag/shadow_campaign.sh <ckpt1.pt> <ckpt2.pt>   # 显式指定
#
# 本地自测（不碰 cargo/serve/Xyce）：
#   bash scripts/diag/shadow_campaign.sh --dry-root /tmp/camp --with-best
#
# 监控：tail -f ~/shadow_campaign.log

set -u

DIAG_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNDIR="${SHADOW_RUNDIR:-/home/tianlang/project-107-v2nowave42m4}"
SCALER="${SHADOW_SCALER:-$RUNDIR/outputs/scaler.pkl}"
PORT="${SHADOW_PORT:-8000}"
ARCHIVE_ROOT="${SHADOW_ARCHIVE_ROOT:-$HOME/shadow_archive}"
FAILED_ROOT="${SHADOW_FAILED_ROOT:-$HOME/shadow_archive_failed}"
TREE="${SHADOW_TREE:-$HOME/NetlistOpt/temp_sim_test/tl_opt_batch}"
ANALYZE_OUT="${SHADOW_ANALYZE_OUT:-$HOME/shadow_analyze.out}"
OUT_DIR="${SHADOW_OUT_DIR:-$HOME}"
SERVE_LOG_DIR="${SHADOW_SERVE_LOG_DIR:-$HOME}"
LOG="${SHADOW_LOG:-$HOME/shadow_campaign.log}"
MAXWAIT="${SHADOW_MAXWAIT:-21600}"
MINFREE_GB="${SHADOW_MINFREE_GB:-50}"
READY_TRIES="${SHADOW_READY_TRIES:-90}"     # × 2s = 最长 3 分钟等 serve 就绪
DRY=0
CKPTS=()
SKIP=()
WITH_BEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY=1 ;;
    --dry-root) DRY=1; R="$2"; TREE="$R/tree"; ARCHIVE_ROOT="$R/archive"
                FAILED_ROOT="$R/failed"; LOG="$R/campaign.log"; ANALYZE_OUT="$R/shadow_analyze.out"
                OUT_DIR="$R"; SERVE_LOG_DIR="$R"; shift ;;
    --rundir)   RUNDIR="$2"; SCALER="${SHADOW_SCALER:-$RUNDIR/outputs/scaler.pkl}"; shift ;;
    --scaler)   SCALER="$2"; shift ;;
    --skip)     SKIP+=("$2"); shift ;;
    --with-best) WITH_BEST=1 ;;
    --maxwait)  MAXWAIT="$2"; shift ;;
    --minfree)  MINFREE_GB="$2"; shift ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    -*)         echo "未知选项: $1"; exit 2 ;;
    *)          CKPTS+=("$1") ;;
  esac
  shift
done

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
tag_of() { basename "$1" .pt; }

# 读**被归档树自己**的 RUN_INFO 拿标签；读不到 → unknown。
# 规则与 run_shadow_batch.sh 的归档块一致（收尾归档最后一棵树时复用同一判据）。
tree_tag() {
  local c
  c=$(sed -n 's/^本轮 serve ckpt *: *//p' "$TREE/RUN_INFO.txt" 2>/dev/null | head -1)
  if [ -n "$c" ]; then basename "$c" .pt; else echo unknown; fi
}

# 归档一棵树到 ARCHIVE_ROOT/<tag>_<时间戳>（与 run_shadow_batch.sh 同一命名规则）
archive_tree() {
  local tag="$1" ts d n
  [ -d "$TREE" ] || return 0
  tag="${tag:-unknown}"; [ -n "$tag" ] || tag=unknown
  ts=$(date +%Y%m%d_%H%M%S); d="$ARCHIVE_ROOT/${tag}_$ts"
  mkdir -p "$ARCHIVE_ROOT"; n=1
  while [ -e "$d" ]; do d="$ARCHIVE_ROOT/${tag}_${ts}_$n"; n=$((n + 1)); done
  mv "$TREE" "$d" && say "  已归档 → $d"
}

# 隔离失败的树：挪出分析池，绝不进 ARCHIVE_ROOT
quarantine_tree() {
  local tag="$1" ts d
  [ -d "$TREE" ] || return 0
  mkdir -p "$FAILED_ROOT"; ts=$(date +%Y%m%d_%H%M%S); d="$FAILED_ROOT/${tag}_$ts"
  mv "$TREE" "$d" && say "  ⚠ 已隔离（不进分析池）→ $d"
  [ -s "$ANALYZE_OUT" ] && mv "$ANALYZE_OUT" "$d/analyze.out" 2>/dev/null
  return 0
}

start_serve() {
  local ckpt="$1" tag="$2" i=0 n body
  if [ "$DRY" = 1 ]; then echo "  [dry] 假装重启 serve --ckpt $ckpt --scaler $SCALER"; return 0; fi
  pkill -f 'serve_htt[p]' 2>/dev/null
  sleep 3
  ( cd "$HOME/-project" && nohup "$HOME/venv/bin/python3" scripts/diag/serve_http.py \
      --ckpt "$ckpt" --scaler "$SCALER" --port "$PORT" \
      > "$SERVE_LOG_DIR/serve_$tag.log" 2>&1 & )
  while [ "$i" -lt "$READY_TRIES" ]; do
    body=$(curl -s -m 2 "http://127.0.0.1:$PORT/" 2>/dev/null || true)
    if printf '%s' "$body" | grep -q '"status"'; then
      n=$(printf '%s' "$body" | sed -n 's/.*"n_models": *\([0-9]*\).*/\1/p')
      if [ "$n" = "1" ]; then echo "  serve 就绪（n_models=1）"; return 0; fi
      say "  ✗ n_models=$n ≠ 1 —— --ckpt 传多了或加载异常，拒绝用这份 serve 起扫"
      return 1
    fi
    i=$((i + 1)); sleep 2
  done
  say "  ✗ serve 就绪超时（${READY_TRIES}×2s）；见 $SERVE_LOG_DIR/serve_$tag.log"
  return 1
}

run_sweep() {
  local ckpt="$1"
  if [ "$DRY" = 1 ]; then
    # 先按 run_shadow_batch.sh 的次序归档上一棵树（否则链式命名这条核心路径没被测到），
    # 再造一棵像样的假树（46 个 CSV），好让下面的校验与归档路径**真的**被执行到
    [ -d "$TREE" ] && archive_tree "$(tree_tag)"
    mkdir -p "$TREE"
    printf '时间            : %s\n本轮 serve ckpt : %s\n' "$(date '+%F %T')" "$ckpt" > "$TREE/RUN_INFO.txt"
    # SHADOW_DRY_NCSV 默认 46（真实电路数）。设小了就是在注入「半趟」故障，
    # 用来验证校验闸门会不会把它拦下并隔离 —— 这是无人值守最要紧的一条防线。
    for i in $(seq 1 "${SHADOW_DRY_NCSV:-46}"); do
      mkdir -p "$TREE/level$((i % 4))/CELL$i"
      echo "eval_idx=1, iter=0, window=0, gnn_pred=1.0e0, true_delay=1.0e-9, transistors=6" \
        > "$TREE/level$((i % 4))/CELL$i/gnn_shadow.csv"
    done
    printf '[%s] 全部分片结束\n候选集数（≥4 候选）: 3   成功行=7 失败行=0 小集=1\n' \
      "$(date '+%F %T')" > "$ANALYZE_OUT"
    return 0
  fi
  bash "$DIAG_DIR/run_shadow_batch.sh"
}

wait_sweep() {
  local t0; t0=$(date +%s)
  if [ "$DRY" = 1 ]; then return 0; fi
  # 这个 pgrep 会同时命中分片与收尾守卫（守卫命令行内含 tl_opt_shadow_batch 字面量），
  # 而守卫活到分析器跑完才退出 → 本循环天然等于「等到分析都写完」。
  while pgrep -f 'tl_opt_shadow_batc[h]' >/dev/null 2>&1; do
    sleep 20
    if [ $(( $(date +%s) - t0 )) -gt "$MAXWAIT" ]; then
      say "  ✗ 单趟超过 ${MAXWAIT}s，杀分片"
      pkill -f 'tl_opt_shadow_batc[h]' 2>/dev/null
      sleep 3
      return 1
    fi
  done
  return 0
}

# ---------- 组 ckpt 清单 ----------
if [ ${#CKPTS[@]} -eq 0 ]; then
  while IFS= read -r p; do [ -n "$p" ] && CKPTS+=("$p"); done \
    < <(ls -1 "$RUNDIR"/outputs/midpoint_ep*.pt 2>/dev/null | sort -V)
  if [ "$WITH_BEST" = 1 ] && [ -f "$RUNDIR/outputs/best_model.pt" ]; then
    CKPTS+=("$RUNDIR/outputs/best_model.pt")
  fi
fi
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "✗ 没有 ckpt 可跑。用 --rundir 指定，或直接传 ckpt 路径。"; exit 2
fi

FINAL=()
for c in "${CKPTS[@]}"; do
  t=$(tag_of "$c")
  hit=0
  for s in ${SKIP[@]+"${SKIP[@]}"}; do [ "$t" = "$s" ] && hit=1; done
  [ "$hit" = 0 ] && FINAL+=("$c")
done
if [ ${#FINAL[@]} -eq 0 ]; then echo "✗ 全部被 --skip 掉了。"; exit 2; fi

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say "===== shadow 战役开始（dry=$DRY）====="
say "rundir  = $RUNDIR"
say "scaler  = $SCALER"
say "tree    = $TREE"
say "归档根  = $ARCHIVE_ROOT（失败隔离 = $FAILED_ROOT）"
say "单趟上限= ${MAXWAIT}s   磁盘下限= ${MINFREE_GB}G"
say "ckpt 清单（${#FINAL[@]} 个）："
for c in "${FINAL[@]}"; do say "  - $(tag_of "$c")   $c"; done

# ---------- 主循环 ----------
for ckpt in "${FINAL[@]}"; do
  tag=$(tag_of "$ckpt"); t0=$(date +%s)
  say "----- [$tag] 开始 -----"

  avail=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$avail" ] && [ "$avail" -lt "$MINFREE_GB" ]; then
    say "  中止战役：磁盘剩余 ${avail}G < ${MINFREE_GB}G"; break
  fi

  if ! start_serve "$ckpt" "$tag"; then say "  serve 未就绪，跳过 $tag"; continue; fi
  say "  起扫（本趟之前那棵树会被自动归档并带上它的 ckpt 标签）"
  run_sweep "$ckpt"
  if ! wait_sweep; then quarantine_tree "$tag"; continue; fi

  ok=1
  n=$(ls -1 "$TREE"/*/*/gnn_shadow.csv 2>/dev/null | wc -l)
  [ "$n" = 46 ] || { say "  ✗ CSV 数=$n（应 46），本趟不完整"; ok=0; }
  [ -s "$ANALYZE_OUT" ] || { say "  ✗ 分析输出缺失或为空"; ok=0; }
  if [ "$ok" = 0 ]; then quarantine_tree "$tag"; continue; fi

  if ! mv "$ANALYZE_OUT" "$OUT_DIR/sweep2_$tag.out"; then
    # 数据本身是好的（46 个 CSV 已校验），只是摘要没转存成 —— 不隔离，
    # 但必须喊出来：下一趟会覆盖 $ANALYZE_OUT，这份读数就只能回头用归档树补算。
    say "  ✗ 转存分析输出失败（$ANALYZE_OUT 会被下一趟覆盖；可事后 --root 归档树补算）"; ok=0
  fi
  rows=$(cat "$TREE"/*/*/gnn_shadow.csv 2>/dev/null | wc -l)
  if [ "$ok" = 1 ]; then
    say "  完成 $tag：行=$rows 用时=$(( $(date +%s) - t0 ))s"
  else
    say "  ⚠ $tag 数据已归档可用，但本趟有错（见上），用时=$(( $(date +%s) - t0 ))s"
  fi
  # 分析器那句元信息原样记下来（候选集数/成功行/失败行），不去 sed 解析它
  grep -m1 '候选集数' "$OUT_DIR/sweep2_$tag.out" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
done

# ---------- 收尾：把最后一棵树也归档 ----------
# run_shadow_batch.sh 只能在「起下一趟」时归档上一棵，所以最后一棵会留在活路径里。
# 这里用同一判据（读树自己的 RUN_INFO）补归档，否则 ~/shadow_archive 会少最后一个点。
if [ -d "$TREE" ]; then
  say "收尾归档最后一棵树"
  archive_tree "$(tree_tag)"
fi

say "===== 战役结束 ====="
{
  echo "归档一览（$ARCHIVE_ROOT）："
  ls -1 "$ARCHIVE_ROOT" 2>/dev/null | sed 's/^/  /'
  echo "隔离一览（$FAILED_ROOT）："
  ls -1 "$FAILED_ROOT" 2>/dev/null | sed 's/^/  /'
} | tee -a "$LOG" | sed 's/^/  /'
