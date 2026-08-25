#!/bin/bash
# 用法: bash setup_exp.sh <base|rank|lib|pgd|pgs|pgs2|struct*|v2wave42|v2nowave42|...>
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
if [ "$V" = "seed789" ]; then           # TRAIN_SEED=789 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 789/' config.py
fi
if [ "$V" = "seed1357" ]; then          # TRAIN_SEED=1357 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 1357/' config.py
fi
if [ "$V" = "seed2468" ]; then          # TRAIN_SEED=2468 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 2468/' config.py
fi
if [ "$V" = "seed3579" ]; then          # TRAIN_SEED=3579 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 3579/' config.py
fi
if [ "$V" = "seed9012" ]; then          # TRAIN_SEED=9012 集成
  sed -i 's/^TRAIN_SEED = .*/TRAIN_SEED = 9012/' config.py
fi
if [ "$V" = "struct" ]; then            # 结构先验特征(transistor_count+门类型计数)
  sed -i 's/^USE_STRUCT_PRIOR = .*/USE_STRUCT_PRIOR = True/' config.py
fi
# STRUCT_MODE 变体（14.2.3：逻辑类别+结构特征替代 638 类 cell 名嵌入）
if [ "$V" = "structbase" ]; then         # 10逻辑 + n_transistors（主测，=默认）
  sed -i "s/^STRUCT_MODE = .*/STRUCT_MODE = 'base'/" config.py
fi
if [ "$V" = "structlogic" ]; then        # 只 10逻辑，无 n_t（消融：n_t 有无贡献）
  sed -i "s/^STRUCT_MODE = .*/STRUCT_MODE = 'logic_only'/" config.py
fi
if [ "$V" = "structrich" ]; then         # +stack +parallel（更多结构细节）
  sed -i "s/^STRUCT_MODE = .*/STRUCT_MODE = 'rich'/" config.py
fi
if [ "$V" = "structelec" ]; then         # p/g/drive 改从 ASAP7 结构算（修 p/g 正则失效）
  sed -i "s/^STRUCT_MODE = .*/STRUCT_MODE = 'elec'/" config.py
fi
# structrich / structlogic 的 seed 变体（补多 seed 确认哪个 cell 策略更好）
# 用法：structrich123 / structlogic2468 等，后缀即 TRAIN_SEED
case "$V" in
  structrich123|structrich2024|structrich456|structrich789|structrich1357|structrich2468|structrich3579|structrich9012)
    sed -i "s/^STRUCT_MODE = .*/STRUCT_MODE = 'rich'/" config.py
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${V#structrich}/" config.py ;;
  structlogic123|structlogic2024|structlogic456|structlogic789|structlogic1357|structlogic2468|structlogic3579|structlogic9012)
    sed -i "s/^STRUCT_MODE = .*/STRUCT_MODE = 'logic_only'/" config.py
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${V#structlogic}/" config.py ;;
esac
if [ "$V" = "waverich" ]; then          # 晶体管波形丰富聚合(mean+max+std)
  sed -i 's/^WAVE_AGG_RICH = .*/WAVE_AGG_RICH = True/' config.py
fi
if [ "$V" = "cornerattn" ]; then        # Corner注意力池化
  sed -i 's/^USE_CORNER_ATTN = .*/USE_CORNER_ATTN = True/' config.py
fi
# V2 数据 wave/no-wave 变体（15.1.x：USE_V2=True 默认生效 + logic_only 自动）
# 用法：v2wave42 / v2nowave123 等，后缀即 TRAIN_SEED；v2nowave 关 wave（Rust 推理拿不到 wave 的验证）
case "$V" in
  v2wave[0-9]*)
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${V#v2wave}/" config.py ;;
  v2nowave[0-9]*)
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${V#v2nowave}/" config.py ;;
esac

# 蒸馏变体（15.2，docs/DISTILL_PLAN.md）：v2kd<teacher><mode><seed>
#   teacher: 123=wave123 | ENS=wave42+123 平均；mode: reg(纯软标签) | rr(reg+rank)
# 例: v2kd123reg42 / v2kd123rr123 / v2kdENSrr42
case "$V" in
  v2kd[0-9]*|v2kdENS*)
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    sed -i "s/^KD_ENABLED = .*/KD_ENABLED = True/" config.py
    case "$V" in
      v2kdENSrr*)  sed -i "s/^KD_MODE = .*/KD_MODE = 'reg+rank'/" config.py; _S="${V#v2kdENSrr}" ;;
      v2kdENSreg*) sed -i "s/^KD_MODE = .*/KD_MODE = 'reg'/" config.py;      _S="${V#v2kdENSreg}" ;;
      v2kd123rr*)  sed -i "s/^KD_MODE = .*/KD_MODE = 'reg+rank'/" config.py; _S="${V#v2kd123rr}" ;;
      v2kd123reg*) sed -i "s/^KD_MODE = .*/KD_MODE = 'reg'/" config.py;      _S="${V#v2kd123reg}" ;;
      *) echo "未知 v2kd 变体: $V（应为 v2kd123reg|v2kd123rr|v2kdENSreg|v2kdENSrr + seed）"; exit 1 ;;
    esac
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
esac

sed -i "s/CACHE_DIR = .*/CACHE_DIR = \"cache107$V\"/" config.py

ulimit -n 8192
# 蒸馏变体默认 teacher 预测目录（可被 KD_TEACHER_DIR 环境变量覆盖）
if [[ "$V" == v2kd* ]]; then
  export KD_TEACHER_DIR="${KD_TEACHER_DIR:-$HOME/project-107-v2wave123/outputs}"
fi
OMP_NUM_THREADS=6 nohup ~/venv/bin/python3 -u main.py > "train107$V.log" 2>&1 &
echo "launched 107-$V  pid=$!  dir=$D"
