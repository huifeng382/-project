#!/bin/bash
# 用法: bash setup_exp.sh <base|pgd|pgs|pgs2>
# 基于 10.7 分支起一个 per_gate 实验，noWave + 独立缓存，后台训练。
set -e
V="$1"
URL="https://github.com/huifeng382/-project.git"
BR="10.3.3-fix-earlystop"
D="$HOME/project-107-$V"

if [ -z "$V" ]; then echo "用法: bash setup_exp.sh <base|pgd|pgs|pgs2>"; exit 1; fi

rm -rf "$D"
git clone -b "$BR" "$URL" "$D"
cd "$D"

# noWave（去掉加载列表里的 batch_wave）
sed -i "s/, 'batch_wave'//" src/train_sweep.py

# per_gate 变体：叠加 10.4 的浅层逐门 loss + node_pred 头
if [ "$V" != "base" ]; then
  git cherry-pick --no-commit ed49d20
fi
# out_slew 变体：把监督目标从 delay 换成 out_slew（100% 密）
if [ "$V" = "pgs" ] || [ "$V" = "pgs2" ]; then
  sed -i 's/per_gate_delay/per_gate_out_slew/g' src/train_sweep.py
fi
# 权重 ×4
if [ "$V" = "pgs2" ]; then
  sed -i 's/+ 0.5 \* F.mse_loss/+ 2.0 * F.mse_loss/' src/train_sweep.py
fi

sed -i "s/CACHE_DIR = .*/CACHE_DIR = \"cache107$V\"/" config.py

OMP_NUM_THREADS=6 nohup python3 -u main.py > "train107$V.log" 2>&1 &
echo "launched 107-$V  pid=$!  dir=$D"
