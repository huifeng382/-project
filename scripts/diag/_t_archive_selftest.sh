#!/bin/bash
# _t_archive_selftest.sh — 自检 run_shadow_batch.sh 的归档/打标逻辑。
# 本地跑，不需要 cargo / serve / Xyce。从真实脚本里**抽出**归档代码块来跑，
# 测的是发布出去的那段文本本身，不是它的复制品。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/run_shadow_batch.sh"
FAIL=0
ok() { echo "  OK   $1"; }
no() { echo "  FAIL $1"; FAIL=$((FAIL + 1)); }

# 抽取归档块。起止模式都**锚行首**：不锚的话，正文注释里只要出现结束模式那几个字，
# awk 就会当场截断 —— 实测踩过（2026-09-18：注释里引用了一次结束模式，抽出的块只剩
# 3 行，6 项自检红 9 条，看起来像功能回归，其实是抽错了）。
BLOCK=$(awk '/^ARCHIVE_ROOT=/{f=1} f{print} /^echo .*本轮 RUN_INFO 已写入/{f=0}' "$SCRIPT")
if [ -z "$BLOCK" ]; then echo "FAIL 抽不到归档代码块"; exit 1; fi
echo "抽取到归档块 $(printf '%s\n' "$BLOCK" | wc -l) 行"
printf 'set -u\n%s\n' "$BLOCK" > /tmp/_blk.sh

# run <沙箱HOME> [ARCHIVE_TAG]
run() {
  local d="$1"
  ( cd "$d/NetlistOpt" || exit 9
    if [ -n "${2:-}" ]; then
      HOME="$d" NL="$d/NetlistOpt" ARCHIVE_TAG="$2" bash /tmp/_blk.sh
    else
      HOME="$d" NL="$d/NetlistOpt" bash /tmp/_blk.sh
    fi )
}

# 造树：$1=沙箱 $2=ckpt名（空=不写 RUN_INFO）
mktree() {
  local d="$1" ckpt="${2:-}"
  mkdir -p "$d/NetlistOpt/temp_sim_test/tl_opt_batch/level0/AND2"
  echo "eval_idx=1, iter=0, window=0, gnn_pred=1.0e0, true_delay=1.0e-9, transistors=6" \
    > "$d/NetlistOpt/temp_sim_test/tl_opt_batch/level0/AND2/gnn_shadow.csv"
  if [ -n "$ckpt" ]; then
    printf '时间            : 2026-09-16 10:00:00\n本轮 serve ckpt : %s\nrepo rev        : deadbee\n' "$ckpt" \
      > "$d/NetlistOpt/temp_sim_test/tl_opt_batch/RUN_INFO.txt"
  fi
}
archs() { ls -1 "$1/shadow_archive" 2>/dev/null | sed 's/_[0-9]\{8\}_[0-9]\{6\}.*$//' | sort; }

echo "--- 1) 首跑：无旧树 → 跳过归档，并建本轮 RUN_INFO ---"
D=$(mktemp -d); mkdir -p "$D/NetlistOpt/temp_sim_test"
out=$(run "$D")
case "$out" in *"无旧树，跳过归档"*) ok "无旧树时跳过归档" ;; *) no "首跑未跳过归档: $out" ;; esac
[ -f "$D/NetlistOpt/temp_sim_test/tl_opt_batch/RUN_INFO.txt" ] \
  && ok "本轮 RUN_INFO 已建" || no "本轮 RUN_INFO 未建"

echo "--- 2) 树内带 RUN_INFO(ep250) → 归档目录名应为 ep250_* ---"
D=$(mktemp -d); mktree "$D" ep250
out=$(run "$D")
got=$(archs "$D")
[ "$got" = "ep250" ] && ok "标签取自树内 RUN_INFO（ep250）" || no "标签错，得到 '$got'"
case "$out" in *"来源=树内RUN_INFO"*) ok "报告了标签来源" ;; *) no "未报告来源: $out" ;; esac

echo "--- 3) ARCHIVE_TAG 覆盖（给无 RUN_INFO 的老树手工命名）---"
D=$(mktemp -d); mktree "$D" ""
out=$(run "$D" sweep_v2nowave42m4_ep250)
got=$(archs "$D")
[ "$got" = "sweep_v2nowave42m4_ep250" ] && ok "ARCHIVE_TAG 生效" || no "ARCHIVE_TAG 未生效，得到 '$got'"

echo "--- 4) 树内无 RUN_INFO 且无 ARCHIVE_TAG → 必须 unknown，且补写说明(不得填本轮 ckpt) ---"
D=$(mktemp -d); mktree "$D" ""
out=$(run "$D")
got=$(archs "$D")
[ "$got" = "unknown" ] && ok "无来源时落 unknown（不猜）" || no "应为 unknown，得到 '$got'"
stub=$(cat "$D/shadow_archive/"*/RUN_INFO.txt 2>/dev/null)
case "$stub" in *"ckpt 未知"*) ok "归档内补写了「ckpt 未知」说明" ;; *) no "补写说明缺失: $stub" ;; esac
case "$stub" in *"未识别"*) no "归档说明里混进了本轮 ckpt（错标复发）" ;; *) ok "未把本轮 ckpt 填进旧树档案" ;; esac

echo "--- 5) 链式三轮：标签必须随数据走，不得 off-by-one ---"
# ⚠ 每轮的树都必须**带 gnn_shadow.csv**：17.4.0 起「无 CSV 的树」不再归档（它只是被 mkdir
#   过、不是一趟数据）。真实流程里被归档的树永远有数据 —— 旧 fixture 没建 CSV，是在测一个
#   现实中不存在的形状，会让新守卫看起来像回归。
D=$(mktemp -d); mkdir -p "$D/NetlistOpt/temp_sim_test/tl_opt_batch/level0/AND2"
putcsv() { mkdir -p "$D/NetlistOpt/temp_sim_test/tl_opt_batch/level0/AND2"
           echo "eval_idx=1, iter=0, window=0, gnn_pred=1.0e0, true_delay=1.0e-9, transistors=6" \
             > "$D/NetlistOpt/temp_sim_test/tl_opt_batch/level0/AND2/gnn_shadow.csv"; }
putcsv
printf '本轮 serve ckpt : ep250\n' > "$D/NetlistOpt/temp_sim_test/tl_opt_batch/RUN_INFO.txt"
run "$D" >/dev/null            # 轮1：归档 ep250 树，写自己的 ckpt(本地取不到→未识别)
putcsv
printf '本轮 serve ckpt : ep150\n' > "$D/NetlistOpt/temp_sim_test/tl_opt_batch/RUN_INFO.txt"
run "$D" >/dev/null            # 轮2：归档的应是 ep150 那棵
got=$(archs "$D")
# 期望恰好两个归档：轮1 归档产出它之前那棵(ep250)，轮2 归档轮1 之后那棵(ep150)。
# off-by-one 的错法会得到 unknown/unknown（用本轮 serve 命名）或少一个。
want=$(printf 'ep150\nep250')
if [ "$got" = "$want" ]; then
  ok "链式标签正确：轮1归档 ep250、轮2归档 ep150，标签随数据走"
else
  no "链式标签 off-by-one，得到: $(printf '%s' "$got" | tr '\n' ' ') / 期望: $(printf '%s' "$want" | tr '\n' ' ')"
fi

echo "--- 6) 空树（有 RUN_INFO、无 gnn_shadow.csv）→ 不得归档（17.4.0 防 junk unknown_*）---"
D=$(mktemp -d); mkdir -p "$D/NetlistOpt/temp_sim_test/tl_opt_batch"
printf '本轮 serve ckpt : ep999\n' > "$D/NetlistOpt/temp_sim_test/tl_opt_batch/RUN_INFO.txt"
out=$(run "$D")
got=$(archs "$D")
[ -z "$got" ] && ok "空树未归档（不造 junk 存档）" || no "空树被归档成了 '$got'"
case "$out" in *"不是一趟数据"*) ok "说明了跳过原因" ;; *) no "未说明跳过原因: $out" ;; esac

echo "--- 7) 变体：TREE 指向 gnn 副本树 → 归档它，且不碰原件树（变体兼容）---"
# 直接测本次要的兼容性：两棵树同时在盘上，块只应搬走 TREE 指的那棵。
D=$(mktemp -d)
mk2() { # $1=树名 $2=ckpt $3=数据文件名（默认 gnn_shadow.csv；内容形状随文件名走，别造假的形状）
  local csv="${3:-gnn_shadow.csv}"
  mkdir -p "$D/NetlistOpt/temp_sim_test/$1/level0/AND2"
  if [ "$csv" = gnn_only.csv ]; then
    # GNN-only 模式的真实形状：**9 列**，且**没有 true_delay 列**（候选根本没跑仿真）。
    # 表头逐字取自 src/gnn_only.rs::new 里那行 writeln!（pred_baseline/ratio/anchored_avg_delay
    # 是锚定设计加的）。夹具形状必须与真实文件一致，否则这个守卫测的是个不存在的形状。
    printf 'eval_idx,iter,window,cand_id,gnn_pred,pred_baseline,ratio,anchored_avg_delay,transistor_count\n2,1,0,e2,1.0e-11,1.2e-11,8.333333e-1,9.600000e-10,6\n' \
      > "$D/NetlistOpt/temp_sim_test/$1/level0/AND2/$csv"
  else
    echo "eval_idx=1, iter=0, window=0, gnn_pred=1.0e0, true_delay=1.0e-9, transistors=6" \
      > "$D/NetlistOpt/temp_sim_test/$1/level0/AND2/$csv"
  fi
  printf '时间            : 2026-09-16 10:00:00\n本轮 serve ckpt : %s\n' "$2" \
    > "$D/NetlistOpt/temp_sim_test/$1/RUN_INFO.txt"
}
mk2 tl_opt_gnn_batch ep250
mk2 tl_opt_batch     ep150
( cd "$D/NetlistOpt" && HOME="$D" NL="$D/NetlistOpt" TREE=temp_sim_test/tl_opt_gnn_batch \
    bash /tmp/_blk.sh ) >/dev/null
got=$(archs "$D")
[ "$got" = "ep250" ] && ok "gnn 树被归档，标签取自它自己的 RUN_INFO" \
  || no "gnn 树归档标签错，得到 '$got'（期望 ep250）"
[ -d "$D/NetlistOpt/temp_sim_test/tl_opt_batch" ] && ok "原件树未被触碰（两变体隔离）" \
  || no "原件树被误归档/误删 —— 变体没隔离住"
[ -f "$D/NetlistOpt/temp_sim_test/tl_opt_gnn_batch/RUN_INFO.txt" ] && ok "gnn 树本轮 RUN_INFO 已重建" \
  || no "gnn 树本轮 RUN_INFO 未建"

echo "--- 8) 变体：TREE 指向 gnnonly 树（数据文件是 gnn_only.csv）→ 也必须归档 ---"
# GNN-only 模式候选没跑仿真，落的是 gnn_only.csv（无 true_delay 列），**不是** gnn_shadow.csv。
# 守卫若只认 gnn_shadow.csv，这棵树会被判成"不是一趟数据"→ 不归档 → 留在原地，
# 下一趟 append 进同一棵 = 混趟。本例如实造出 gnnonly 的真实形状来测守卫放宽了没有。
D=$(mktemp -d)
mk2 tl_opt_gnn_only  ep250 gnn_only.csv
mk2 tl_opt_gnn_batch ep150 gnn_shadow.csv
( cd "$D/NetlistOpt" && HOME="$D" NL="$D/NetlistOpt" TREE=temp_sim_test/tl_opt_gnn_only \
    bash /tmp/_blk.sh ) >/dev/null
got=$(archs "$D")
[ "$got" = "ep250" ] && ok "gnnonly 树（gnn_only.csv）被归档，标签取自它自己的 RUN_INFO" \
  || no "gnnonly 树归档标签错，得到 '$got'（期望 ep250）—— 守卫可能仍只认 gnn_shadow.csv"
[ -d "$D/NetlistOpt/temp_sim_test/tl_opt_gnn_batch" ] && ok "gnn 对照树未被触碰（两变体隔离）" \
  || no "gnn 对照树被误归档/误删 —— 变体没隔离住"
[ -f "$D/NetlistOpt/temp_sim_test/tl_opt_gnn_only/RUN_INFO.txt" ] && ok "gnnonly 树本轮 RUN_INFO 已重建" \
  || no "gnnonly 树本轮 RUN_INFO 未建"

echo "=================================================="
if [ "$FAIL" -gt 0 ]; then echo "FAIL 共 $FAIL 项"; exit 1; fi
echo "OK 全部通过"
