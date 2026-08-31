#!/bin/bash
# run_shadow_batch.sh — Rust 46 候选 shadow 基准标准启动（16.9.0，防两个坑）
#
# 坑1（慢）：批次默认串行 + Xyce 单核 → 24 核只用一个核，46 电路排队数小时。
#   对策：按 level 分片（level0-3 各一进程）+ level4 按电路逐个进程，全并行。
# 坑2（错）：gnn_shadow.csv 是 append 模式且路径固定（temp_sim_test/tl_opt_batch），
#   旧 run 的行会混入 → 分析结果错误（曾混入 8/27 旧模型 ~1100 行）。
#   对策：启动前强制 rm -rf 该目录（必须在分片启动前做一次，不能每个分片各清一次）。
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

# 1) 清理旧 CSV（防污染——append 模式，旧行混入会让分析结果错）
rm -rf temp_sim_test/tl_opt_batch
echo "[$(date +%F\ %T)] 已清理 temp_sim_test/tl_opt_batch（防旧 run CSV 混入）"

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
