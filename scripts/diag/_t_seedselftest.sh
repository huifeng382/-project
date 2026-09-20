#!/usr/bin/env bash
# setup_exp.sh 数据播种块（18.2.0 改）的隔离自检。
#
# 做法：用 sed 从真实 setup_exp.sh 里抽出那段 if，配好变量在沙箱里跑 —— 测的是**真实文本**，
# 不是抄一遍的副本。四个场景：
#   S1 V3，种子源含 v3_delivery        → 应从 CACHE_SEED/data 复制，且 mtime 保住
#   S2 V3，种子源只有 V2（无 v3_delivery）→ 应退化到 $HOME/-project/data 并复制
#   S3 V2，DATA_BATCHES=旧三批          → 应复制旧四批（与旧行为一致）
#   S4 V3 + RESUME=1 且数据齐全         → 应跳过播种
# 用法: bash scripts/diag/_t_seedselftest.sh
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO/setup_exp.sh"
T=/tmp/seedselftest
PASS=0; FAIL=0

BLK="$T/block.sh"
mkdir -p "$T"
# 按**结构**抽：从外层 `if [ -n "$CACHE_SEED" ]` 抽到第一个顶格 `fi`（内层 if 都是缩进的）。
# ⚠ 别用硬编码行号 —— 第一次就是抽到了内层 `fi` 导致整块语法残缺、15 项全挂，
#   而「末行是 fi」这种形状检查根本拦不住（缩进的 fi 也长这样）。
awk '/^if \[ -n "\$CACHE_SEED" \]/{f=1} f{print} f&&/^fi$/{exit}' "$SRC" > "$BLK"
# 抽出来的必须确实是那段：首行是外层 if，末行是**顶格** fi
head -1 "$BLK" | grep -q 'CACHE_SEED' || { echo "FATAL: 抽取起点不对"; exit 1; }
tail -1 "$BLK" | grep -qx 'fi'        || { echo "FATAL: 抽取终点不是顶格 fi"; exit 1; }
bash -n "$BLK"                        || { echo "FATAL: 抽出的块语法不通"; exit 1; }

ok()   { PASS=$((PASS+1)); echo "    ✅ $1"; }
bad()  { FAIL=$((FAIL+1)); echo "    ❌ $1"; }
chk()  { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: 期望 $3 实得 $2"; fi; }

# 造种子源：V3 集 + 旧四批，mtime 打成一个可辨识的时间
mkseed() {                       # $1=根目录  $2=要造的集（空格分隔）
  rm -rf "$1"; mkdir -p "$1"
  for b in $2; do
    mkdir -p "$1/data/$b"
    echo "seed-$b" > "$1/data/$b/circuit_static.parquet"
    touch -t 202001020304 "$1/data/$b/circuit_static.parquet"
  done
}

echo "== 播种块自检（4 场景）=="

# ---------------- S1：V3，种子源含 v3_delivery
echo "S1 V3，种子源含 v3_delivery"
mkseed "$T/seed1" "v3_delivery batch_v2_io"
D="$T/tree1"; rm -rf "$D"; mkdir -p "$D/data"
( CACHE_SEED="$T/seed1" D="$D" DATA_BATCHES='v3_delivery' RESUME_MODE=0 HOME="$T/home1" \
  sh "$BLK" ) > "$T/s1.log" 2>&1
chk "v3_delivery 被复制" "$([ -f "$D/data/v3_delivery/circuit_static.parquet" ] && echo yes || echo no)" "yes"
chk "mtime 保住(2020-01-02)" "$(date -r "$D/data/v3_delivery/circuit_static.parquet" +%Y%m%d 2>/dev/null)" "20200102"
grep -q 'WARN' "$T/s1.log" && bad "不该有 WARN" || ok "无 WARN"

# ---------------- S1b：mtime 覆盖的是 clone 出来的旧数据（核心回归）
echo "S1b 树里已有 clone 版 v3_delivery（mtime=今天）→ 应被种子覆盖"
D="$T/tree1b"; rm -rf "$D"; mkdir -p "$D/data/v3_delivery"
echo clone > "$D/data/v3_delivery/circuit_static.parquet"     # mtime = now
( CACHE_SEED="$T/seed1" D="$D" DATA_BATCHES='v3_delivery' RESUME_MODE=0 HOME="$T/home1" \
  sh "$BLK" ) > "$T/s1b.log" 2>&1
chk "mtime 已改成种子值" "$(date -r "$D/data/v3_delivery/circuit_static.parquet" +%Y%m%d)" "20200102"
chk "内容来自种子" "$(cat "$D/data/v3_delivery/circuit_static.parquet")" "seed-v3_delivery"

# ---------------- S2：V3，种子源只有 V2 → 退化到 $HOME/-project/data
echo "S2 V3，种子源缺 v3_delivery → 退化到 \$HOME/-project/data"
mkseed "$T/seed2" "batch_v2_full batch_v2_rest batch_v2_io batch_v2_m4"
mkdir -p "$T/home2"; mkseed "$T/home2/-project" "v3_delivery"   # 本场景专用，避免污染别的场景
D="$T/tree2"; rm -rf "$D"; mkdir -p "$D/data"
( CACHE_SEED="$T/seed2" D="$D" DATA_BATCHES='v3_delivery' RESUME_MODE=0 HOME="$T/home2" \
  sh "$BLK" ) > "$T/s2.log" 2>&1
grep -q '退化到统一数据源' "$T/s2.log" && ok "打印了退化提示" || bad "没打印退化提示"
chk "v3_delivery 从退化源复制" "$([ -f "$D/data/v3_delivery/circuit_static.parquet" ] && echo yes || echo no)" "yes"
chk "mtime 保住" "$(date -r "$D/data/v3_delivery/circuit_static.parquet" +%Y%m%d)" "20200102"

# ---------------- S2b：两处都没有 v3_delivery → 必须 WARN，不能静默
echo "S2b 两处都没有 v3_delivery → 应 WARN（不静默）"
mkdir -p "$T/home2b"                                            # 专用空 HOME：退化源里也没有
D="$T/tree2b"; rm -rf "$D"; mkdir -p "$D/data"
( CACHE_SEED="$T/seed2" D="$D" DATA_BATCHES='v3_delivery' RESUME_MODE=0 HOME="$T/home2b" \
  sh "$BLK" ) > "$T/s2b.log" 2>&1
grep -q 'WARN.*v3_delivery' "$T/s2b.log" && ok "有 WARN" || bad "静默失败了"

# ---------------- S3：V2 变体 → 与旧行为一致（旧四批）
echo "S3 V2 变体（DATA_BATCHES=旧三批）"
mkdir -p "$T/home3"                                             # 专用空 HOME：确保只从种子源取
D="$T/tree3"; rm -rf "$D"; mkdir -p "$D/data"
( CACHE_SEED="$T/seed2" D="$D" DATA_BATCHES='batch_v2_full,batch_v2_rest,batch_v2_m4' \
  RESUME_MODE=0 HOME="$T/home3" sh "$BLK" ) > "$T/s3.log" 2>&1
for b in batch_v2_full batch_v2_rest batch_v2_m4 batch_v2_io; do
  chk "$b 被复制" "$([ -f "$D/data/$b/circuit_static.parquet" ] && echo yes || echo no)" "yes"
done
chk "没多复制 v3_delivery" "$([ -d "$D/data/v3_delivery" ] && echo yes || echo no)" "no"
grep -q 'seeded 4 dataset' "$T/s3.log" && ok "计数=4" || bad "计数不对: $(grep seeded "$T/s3.log")"

# ---------------- S4：RESUME=1 且数据齐全 → 跳过
echo "S4 V3 + RESUME=1 且数据齐全 → 跳过播种"
D="$T/tree4"; rm -rf "$D"; mkdir -p "$D/data"
( CACHE_SEED="$T/seed1" D="$D" DATA_BATCHES='v3_delivery' RESUME_MODE=0 HOME="$T/home1" sh "$BLK" ) >/dev/null 2>&1
echo "tampered" > "$D/data/v3_delivery/circuit_static.parquet"     # 若被覆盖说明没跳过
( CACHE_SEED="$T/seed1" D="$D" DATA_BATCHES='v3_delivery' RESUME_MODE=1 HOME="$T/home1" \
  sh "$BLK" ) > "$T/s4.log" 2>&1
grep -q 'RESUME: 目录已有本次所需数据' "$T/s4.log" && ok "打印跳过" || bad "没跳过"
chk "数据没被覆盖" "$(cat "$D/data/v3_delivery/circuit_static.parquet")" "tampered"

echo
echo "== 结果：$PASS 通过 / $FAIL 失败 =="
[ "$FAIL" -eq 0 ] || exit 1
