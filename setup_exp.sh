#!/bin/bash
# 用法: bash setup_exp.sh <base|rank|lib|pgd|pgs|pgs2>
# 基于 10.7 分支起一个 per_gate 实验，noWave + 独立缓存，后台训练。
set -e
V="$1"
URL="https://github.com/huifeng382/-project.git"
BR="10.3.3-fix-earlystop"
D="$HOME/project-107-$V"

if [ -z "$V" ]; then echo "用法: bash setup_exp.sh <base|rank|lib|pgd|pgs|pgs2>"; exit 1; fi

rm -rf "$D"
git clone -b "$BR" "$URL" "$D"
cd "$D"

# lib 变体：Scheme A（train_lib + SC展开LIB链），QUICK_TEST 先测速
if [ "$V" = "lib" ]; then
  sed -i 's/from src.train_sweep import main/from src.train_lib import main/' main.py
  sed -i "s/, 'batch_wave'//" src/train_lib.py
  sed -i 's/^QUICK_TEST = .*/QUICK_TEST = True/' config.py
  sed -i 's/CACHE_DIR = .*/CACHE_DIR = "cache107lib"/' config.py
  ulimit -n 8192
OMP_NUM_THREADS=6 nohup ~/venv/bin/python3 -u main.py > "train107lib.log" 2>&1 &
  echo "launched 107-lib QUICK_TEST  pid=$!  dir=$D"
  exit 0
fi

# per_gate 变体（pgd/pgs/pgs2）：先在干净树上 cherry-pick 10.4（浅层逐门 loss + node_pred 头）
if [ "$V" = "pgd" ] || [ "$V" = "pgs" ] || [ "$V" = "pgs2" ]; then
  git cherry-pick --no-commit ed49d20
fi

# noWave（去掉加载列表里的 batch_wave）。旧数据在 archive_v13.1/，delivery1 在 data/delivery1/
sed -i "s/, 'batch_wave'//" src/train_sweep.py

# out_slew 变体：把监督目标从 delay 换成 out_slew（100% 密）
if [ "$V" = "pgs" ] || [ "$V" = "pgs2" ]; then
  sed -i 's/per_gate_delay/per_gate_out_slew/g' src/train_sweep.py
fi
# 权重 ×4
if [ "$V" = "pgs2" ]; then
  sed -i 's/+ 0.5 \* F.mse_loss/+ 2.0 * F.mse_loss/' src/train_sweep.py
fi

# 优化探索变体（同一 expr 切分，仅改 config，互相可比）
if [ "$V" = "anneal" ]; then          # 更深退火
  sed -i 's/^LR_MIN = .*/LR_MIN = 1e-7/' config.py
  sed -i 's/^LR_FACTOR = .*/LR_FACTOR = 0.4/' config.py
fi
if [ "$V" = "bmvl" ]; then             # best_model 按 val_loss 选点
  sed -i "s/^BEST_MODEL_METRIC = .*/BEST_MODEL_METRIC = 'val_loss'/" config.py
fi
if [ "$V" = "bmsm" ]; then             # best_model 按平滑 rel_err 选点
  sed -i "s/^BEST_MODEL_METRIC = .*/BEST_MODEL_METRIC = 'smoothed_rel_err'/" config.py
fi
if [ "$V" = "es" ]; then               # 早停放宽（练更久，防欠训）
  sed -i 's/^PATIENCE = .*/PATIENCE = 100/' config.py
  sed -i 's/^PLATEAU_MIN_EPOCHS = .*/PLATEAU_MIN_EPOCHS = 200/' config.py
fi
if [ "$V" = "rankloss1" ]; then         # 成对排序损失 w=0.5
  sed -i 's/^RANK_LOSS_W = .*/RANK_LOSS_W = 0.5/' config.py
fi
if [ "$V" = "rankloss2" ]; then         # 成对排序损失 w=2.0
  sed -i 's/^RANK_LOSS_W = .*/RANK_LOSS_W = 2.0/' config.py
fi
if [ "$V" = "bestrank" ]; then          # checkpoint 按 val 选择遗憾选
  sed -i "s/^BEST_RANK_METRIC = .*/BEST_RANK_METRIC = 'regret'/" config.py
fi
# delivery1 新物理特征消融实验（独立控制，默认全关=纯基线）
if [ "$V" = "newcaps" ]; then           # +parasitic_caps 每门寄生电容
  sed -i 's/^USE_PARASITIC_CAPS = .*/USE_PARASITIC_CAPS = True/' config.py
fi
if [ "$V" = "newwave" ]; then           # +transistor_wave 晶体管波形
  sed -i 's/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = True/' config.py
fi
if [ "$V" = "newnoise" ]; then          # +supply_noise 电源噪声
  sed -i 's/^USE_SUPPLY_NOISE = .*/USE_SUPPLY_NOISE = True/' config.py
fi
if [ "$V" = "seed123" ]; then           # TRAIN_SEED=123 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 123/' config.py
fi
if [ "$V" = "seed2024" ]; then          # TRAIN_SEED=2024 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 2024/' config.py
fi
if [ "$V" = "seed456" ]; then           # TRAIN_SEED=456 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 456/' config.py
fi
if [ "$V" = "struct" ]; then            # 结构先验特征(transistor_count+门类型计数)
  sed -i 's/^USE_STRUCT_PRIOR = .*/USE_STRUCT_PRIOR = True/' config.py
fi
if [ "$V" = "waverich" ]; then          # 晶体管波形丰富聚合(mean+max+std)
  sed -i 's/^WAVE_AGG_RICH = .*/WAVE_AGG_RICH = True/' config.py
fi
if [ "$V" = "cornerattn" ]; then        # Corner注意力池化
  sed -i 's/^USE_CORNER_ATTN = .*/USE_CORNER_ATTN = True/' config.py
fi

sed -i "s/CACHE_DIR = .*/CACHE_DIR = \"cache107$V\"/" config.py

ulimit -n 8192
OMP_NUM_THREADS=6 nohup ~/venv/bin/python3 -u main.py > "train107$V.log" 2>&1 &
echo "launched 107-$V  pid=$!  dir=$D"
