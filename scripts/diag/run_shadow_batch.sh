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
#   归档目录保留 <root>/<level>/<stem>/gnn_shadow.csv 原布局 → **本身就是合法 --root**，
#   可事后直接 `_shadow_analyze.py --root ~/shadow_archive/<TAG>_<时间戳>`。
#   存档不自动清理，占盘自己看情况删（CSV 很小，单趟 MB 级）。
#
# 变体：本脚本可驱动**两套** batch，输出树/报告/日志已各自隔离，可并存互不污染：
#   bash run_shadow_batch.sh          → shadow（原件 tl_opt_shadow_batch，树 temp_sim_test/tl_opt_batch）
#   bash run_shadow_batch.sh gnn      → gnn   （副本 tl_opt_gnn_batch，树 temp_sim_test/tl_opt_gnn_batch）
#   两套的 Rust 参数逐字相同，差别只在输出路径。gnn 副本是 NetlistOpt 16.0.0 为
#   「GNN 取代仿真」准备的 —— ⚠ 现状**仿真并未减少**，见那份文件头部说明。
#   副本自身的默认输出根已等于本脚本的 gnn 树，故**不需要**再设 SHADOW_OUT_BASE。
#   ⚠ shadow_campaign.sh / _shadow_ckpt_sweep.sh 目前都以无参数方式调用本脚本（= shadow）；
#     要用它们驱动 gnn，得先给它们加透传，别只改这里。
#
#   17.3.17：修 17.3.16 的**错标**。17.3.16 在本轮开头归档上一轮的树，却用**本轮 serve 的
#   ckpt** 给归档命名 → 每份存档都错一格（跑 ep150 时归档出的目录叫 ep150，里面其实是
#   0916 那趟的数据）。错得还很像对的，正是 I15 那类「来源不可信」。现改为：
#     ① 本轮开始时把**本轮**的身份写进 $TREE/RUN_INFO.txt；
#     ② 归档时读**被归档树自己**的 RUN_INFO 来命名 —— 标签随数据走，不靠猜；
#     ③ 读不到就写 unknown（`ARCHIVE_TAG` 可手工指定，例如给当前这棵无 RUN_INFO 的老树）；
#     ④ 归档后若树内本来没有 RUN_INFO，补一份说明「ckpt 未知」，而不是填本轮的 ckpt。
#
# 前置：GNN serve 已运行。**必须带下面这组 env**，理由见 shadow_campaign.sh 的 SERVE_ENV 注释：
#   不带 → serve 的 OMP worker 在两批之间空转自旋，实测同一批候选 828% CPU vs 3.3% CPU（248×），
#   纯浪费且会拖慢同机的 zhirui（16 路 Xyce）。这是等待策略、不改计算 → 位级中性、可安全默认开。
#   cd ~/-project && GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \
#     nohup ~/venv/bin/python3 scripts/diag/serve_http.py \
#     --ckpt CKPT路径.pt --scaler SCALER路径.pkl --port 8000 &
#   这一步留一份固定副本：~/NetlistOpt/serve_env.sh（内容即上面那行的 env 段）
# 用法：bash ~/-project/scripts/diag/run_shadow_batch.sh [shadow|gnn]   （缺省 shadow）
# 结果：全部跑完自动执行 _shadow_analyze.py，输出到 ~/shadow_analyze.out
#       （gnn 变体另存 ~/shadow_analyze_gnn.out，免得两趟收尾互相覆盖）

set -u
NL="$HOME/NetlistOpt"
[ -d "$NL" ] || { echo "ERROR: 没有 $NL"; exit 1; }

# 0a) 变体选择（$1）。TREE 一律**无条件**赋值、不读环境变量：防将来有人 export TREE
#     时静默改掉落点（shadow_campaign.sh 自己也有个 TREE，两处语义不同，别互相渗透）。
VARIANT="${1:-shadow}"
case "$VARIANT" in
  shadow) TEST=tl_opt_shadow_batch; SUB=tl_opt_batch ;;
  gnn)    TEST=tl_opt_gnn_batch;    SUB=tl_opt_gnn_batch ;;
  *) echo "ERROR: 未知变体 '$VARIANT'（可用 shadow | gnn）"; exit 1 ;;
esac
TREE="temp_sim_test/$SUB"
LOG_PFX="$HOME/shadow"; ANA_OUT="$HOME/shadow_analyze.out"
if [ "$VARIANT" != shadow ]; then
  LOG_PFX="$HOME/shadow_$VARIANT"; ANA_OUT="$HOME/shadow_analyze_$VARIANT.out"
fi
echo "[$(date +%F\ %T)] 变体=$VARIANT  test=$TEST  树=$TREE  日志前缀=$LOG_PFX"

# 0) 检查 GNN serve（括号技巧防自匹配）
if ! pgrep -f 'serve_htt[p]' >/dev/null; then
  echo "ERROR: GNN serve 未运行。先启动（⚠ 必须带这组 env，否则 248× 空转）："
  echo "  cd ~/-project && GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \\"
  echo "    nohup ~/venv/bin/python3 scripts/diag/serve_http.py --ckpt CKPT路径.pt --scaler SCALER路径.pkl --port 8000 &"
  exit 1
fi
# 0b) 空转自旋自检（17.4.0）：serve 在跑 ≠ serve 没在烧核。这条只告警不拦 ——
#     有意做「带 env / 不带 env」对照实验时仍需放行，但必须让人看见。
SPID=$(pgrep -f 'serve_htt[p]' | head -1)
if [ -n "$SPID" ] && ! tr '\0' '\n' < "/proc/$SPID/environ" 2>/dev/null | grep -q '^GOMP_SPINCOUNT=0$'; then
  echo "⚠⚠ 警告：serve(PID=$SPID) 未带 GOMP_SPINCOUNT=0 —— 它会在两批之间空转自旋。"
  echo "    实测代价 828% CPU vs 3.3%（248×），且会加重与 zhirui 的内存带宽/LLC 争用。"
  echo "    建议停掉重起（按上面的前置命令）。若你是有意做对照，忽略本条即可。"
fi

cd "$NL"

# 1) 归档旧树（防污染——append 模式，旧行混入会让分析结果错；但不再销毁）
#    标签取**被归档树自己**的 RUN_INFO（那描述的是产出这棵树的上一轮），读不到就 unknown。
#    ⚠ 绝不拿本轮 serve 的 ckpt 命名旧树 —— 那是错标，会重演 I15 那类「来源不可信」。
ARCHIVE_ROOT="$HOME/shadow_archive"
#   下面几行是给**被单独抽出的本块**兜底：_t_archive_selftest.sh 用 awk 抽出本块单独运行
#   （起于 ARCHIVE_ROOT=、止于 RUN_INFO 落盘那行 echo），那里没有 $TREE/$VARIANT/$TEST，
#   缺兜底会以「未绑定变量」中断、把自检变成假回归。脚本正常路径上它们已在上方定好。
#   ⚠ awk 的结束模式是那行 echo 的原文，注释里**不要照抄它**：照抄会提前截断抽出的块
#     （实测踩过：抽出的块只剩 3 行，6 项自检红 9 条）。
: "${TREE:=temp_sim_test/tl_opt_batch}" "${VARIANT:=shadow}" "${TEST:=tl_opt_shadow_batch}"
SERVE_ARGS=$(ps -o args= -p "$(pgrep -f 'serve_htt[p]' 2>/dev/null | head -1)" 2>/dev/null | head -1 || true)
CKPT=$(printf '%s\n' "${SERVE_ARGS:-}" | sed -n 's/.*--ckpt[= ][ ]*\([^ ]*\).*/\1/p' | head -1)
# ckpt 身份取**内容 sha1**（不是路径）。同一份权重可能被复制成多个文件名、同一个文件名也可能
# 被覆盖重训 —— 只有 sha1 能回答「报的这个数挂的是不是这个文件」。与 serve 的 `_sha16` 同法。
CKPT_SHA=$( [ -n "${CKPT:-}" ] && sha1sum "$CKPT" 2>/dev/null | cut -c1-16 )
OLD_CKPT=$(sed -n 's/^本轮 serve ckpt *: *//p' "$TREE/RUN_INFO.txt" 2>/dev/null | head -1)
# 17.4.0：只归档**有数据**的树。空的（或只剩 RUN_INFO 的）树归档出来就是一份 junk
# `unknown_<时间戳>` 存档 —— 看着像一趟数据，其实只证明「这个目录被 mkdir 过」。
# 实测踩过（2026-09-17，见 #65）：一条 mkdir -p 命令就造出一份 1 文件的假存档。
TREE_HAVE_CSV=$(ls -1 "$TREE"/*/*/gnn_shadow.csv 2>/dev/null | head -1)
if [ -d "$TREE" ] && [ -n "$TREE_HAVE_CSV" ]; then
  TAG=${ARCHIVE_TAG:-$(basename "${OLD_CKPT:-unknown}" .pt)}; [ -n "$TAG" ] || TAG=unknown
  if [ -n "$OLD_CKPT" ]; then SRC=树内RUN_INFO; else SRC=兜底; fi
  TS=$(date +%Y%m%d_%H%M%S)
  DEST="$ARCHIVE_ROOT/${TAG}_$TS"
  mkdir -p "$ARCHIVE_ROOT"
  n=1
  while [ -e "$DEST" ]; do DEST="$ARCHIVE_ROOT/${TAG}_${TS}_$n"; n=$((n + 1)); done
  if ! mv "$TREE" "$DEST"; then
    echo "ERROR: 归档失败（$DEST）—— 拒绝对未清理的旧 CSV 继续跑（append 会混趟）"
    exit 1
  fi
  if [ ! -f "$DEST/RUN_INFO.txt" ]; then
    printf '本轮 serve ckpt : unknown\n（归档时树内没有 RUN_INFO —— 这棵树的 ckpt 未知；目录标签 %s 来自 ARCHIVE_TAG 或 unknown，不可当已证来源）\n' "$TAG" > "$DEST/RUN_INFO.txt"
  fi
  echo "[$(date +%F\ %T)] 已归档旧树 → $DEST  (标签=$TAG 来源=$SRC)"
elif [ -d "$TREE" ]; then
  echo "[$(date +%F\ %T)] 旧树无 gnn_shadow.csv → 不是一趟数据，跳过归档（不造 junk unknown_* 存档）"
else
  echo "[$(date +%F\ %T)] 无旧树，跳过归档（首跑）"
fi

# 1b) 写**本轮** RUN_INFO —— 随这棵树进归档，供下一轮正确命名（先建目录，分片只管往里写子目录）
#     17.4.0 补：身份不能只记「文件名 / commit 号」。2026-09-17 追 74.4-vs-90.8 时发现，
#     当时能记下的三样（ckpt 路径、repo rev、dirty 个数）**没有一样能唯一确定读数**：
#       · ckpt 路径 ≠ 内容（同名可被覆盖重训）→ 已补 sha1；
#       · `repo rev` 是 **commit**，服务器 ~/-project 停在 16.10.0 却 dirty 41 个文件、脚本
#         是手工同步的新版 → rev 相同 ≠ 脚本相同；
#       · **Rust 树 ~/NetlistOpt 服务器上没有 .git**，vintage 无任何记录，而 gnn_pred 的
#         秩聚合就发生在它的 Rust 侧 → 这是本类争议里唯一无法排除的变量。
#     故改为记**内容指纹**：dirty 指纹（git diff + status 一起哈希）、Rust 源码指纹
#     （无 VCS 就哈希 src/ tests/ Cargo.toml 全文 + 相对路径）、serve 脚本 sha1
#     （gnn_pred = 批内平均秩，由它的 predict_rank_batch 算出来）。
mkdir -p "$TREE"
REPO_DIRTY_FP=$(cd "$HOME/-project" 2>/dev/null && { git diff 2>/dev/null; git status --porcelain 2>/dev/null; } | sha1sum | cut -c1-12)
RUST_FP=$(cd "$NL" 2>/dev/null && find src tests Cargo.toml -type f 2>/dev/null | LC_ALL=C sort \
          | xargs -r sha1sum 2>/dev/null | sha1sum | cut -c1-12)
RUST_REV=$(cd "$NL" 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo 无VCS)
SERVE_PY_FP=$(sha1sum "$HOME/-project/scripts/diag/serve_http.py" 2>/dev/null | cut -c1-12)
{
  echo "时间            : $(date +%F\ %T)"
  echo "变体            : $VARIANT  （cargo --test $TEST，输出树 $TREE）"
  echo "源目录          : $NL/$TREE"
  echo "本轮 serve 进程 : ${SERVE_ARGS:-未取到}"
  echo "本轮 serve ckpt : ${CKPT:-未识别}"
  echo "本轮 ckpt sha1  : ${CKPT_SHA:-未识别}"
  echo "repo rev        : $(cd ~/-project 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo 未知)"
  echo "repo dirty      : $(cd ~/-project 2>/dev/null && git status --porcelain 2>/dev/null | wc -l || echo '?') 个改动"
  echo "repo dirty 指纹 : ${REPO_DIRTY_FP:-未识别}  （git diff + status 的 sha1 前12位）"
  echo "Rust rev        : ${RUST_REV}  （无 VCS 时此项无意义，看下一行）"
  echo "Rust 源指纹     : ${RUST_FP:-未识别}  （src/ tests/ Cargo.toml 全文+路径的 sha1 前12位）"
  echo "serve 脚本指纹  : ${SERVE_PY_FP:-未识别}  （scripts/diag/serve_http.py sha1 前12位）"
} > "$TREE/RUN_INFO.txt"
echo "[$(date +%F\ %T)] 本轮 RUN_INFO 已写入（ckpt=${CKPT:-未识别} sha1=${CKPT_SHA:-未识别}）"

# 2) 并行分片：level0-3 各一个进程；level4 按电路逐个进程（大电路最慢，全并行）
for l in 0 1 2 3; do
  TL_ONLY=level$l GNN_SHADOW=1 GNN_HOST=127.0.0.1 GNN_PORT=8000 SPICEVIZ_OFF=1 \
    nohup cargo test --release --test $TEST -- --nocapture --ignored \
    > ${LOG_PFX}_lvl$l.log 2>&1 &
done
for c in $(ls testbench/tl_cells/level4/*.tl | xargs -n1 basename | sed 's/\.tl$//'); do
  # 16.11.6: level4/ 前缀精确匹配（防 OVF 误带 ADD4_OVF/ovf1 → 并发写同 CSV 损坏）
  TL_ONLY=level4/$c GNN_SHADOW=1 GNN_HOST=127.0.0.1 GNN_PORT=8000 SPICEVIZ_OFF=1 \
    nohup cargo test --release --test $TEST -- --nocapture --ignored \
    > ${LOG_PFX}_lvl4_$c.log 2>&1 &
done
echo "[$(date +%F\ %T)] 已启动并行分片（变体=$VARIANT test=$TEST，level0-3 + level4 每电路一个进程）"

# 3) 自动收尾：所有 cargo 分片结束后跑分析
#    轮询用 cargo 模式（不要锚定二进制哈希——cargo 重编译后哈希会变）
#    轮询模式、--root、输出文件三处都随变体走：两套的 pgrep 模式互不匹配（测试名不同），
#    所以并行跑两个变体时各自的收尾只认自己的分片，不会谁先结束就把对方提前收掉。
CLOSER="while pgrep -f \"car[g]o test --release --test $TEST\" >/dev/null 2>&1; do sleep 30; done; sleep 5; { echo \"[\$(date +%F\\ %T)] 全部分片结束\"; cd \$HOME/-project && \$HOME/venv/bin/python3 scripts/diag/_shadow_analyze.py --root $NL/$TREE; } > $ANA_OUT 2>&1"
nohup bash -c "$CLOSER" > /dev/null 2>&1 &
echo "[$(date +%F\ %T)] 自动收尾已挂（完成后写 $ANA_OUT）"
echo "监控: tail -f $ANA_OUT"
