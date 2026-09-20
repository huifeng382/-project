#!/bin/bash
# 用法: bash setup_exp.sh <base|rank|lib|pgd|pgs|pgs2|struct*|v2wave42|v2nowave42|v2ia42|v2cov2542|...>
# 基于 10.7 分支起一个 per_gate 实验，noWave + 独立缓存，后台训练。
set -e
V="$1"
URL="https://github.com/huifeng382/-project.git"
BR="10.3.3-fix-earlystop"
D="$HOME/project-107-$V"

if [ -z "$V" ]; then echo "用法: bash setup_exp.sh <base|rank|lib|pgd|pgs|pgs2>"; exit 1; fi

# 16.8.0 RESUME 模式：目录已存在且代码完整时跳过 rm/clone（保留图缓存/掩码，增量续建）
# 用法: RESUME=1 CACHE_SEED=<master> bash setup_exp.sh <原变体名>
RESUME_MODE=0
if [ "$RESUME" = "1" ] && [ -f "$D/main.py" ] && [ -f "$D/config.py" ] && [ -d "$D/src" ]; then
  case "$D" in
    *"$V") ;;
    *) echo "ERROR: RESUME 变体名 $V 与目录 $D 不匹配（RESUME 必须用原变体名）"; exit 1 ;;
  esac
  for _p in $(pgrep -f 'main[.]py' 2>/dev/null || true); do
    if [ "$(realpath /proc/$_p/cwd 2>/dev/null)" = "$D" ]; then
      echo "ERROR: $D 已有训练进程在跑（pid=$_p），禁止重复启动"; exit 1
    fi
  done
  echo "RESUME mode: $D 已存在，跳过 rm/clone（缓存增量续建、掩码命中保留）"
  RESUME_MODE=1
else
  rm -rf "$D"
  git clone -b "$BR" "$URL" "$D"
fi
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
# 16.11.4: 支持 v2nowave42m4 / v2iaa42m4 等（尾部可带 m4 标记，seed 取数字前缀）
case "$V" in
  v2nowavegnn[0-9]*)   # 17.1.2: 纯拓扑 nowave + GNN 预测 ids_avg 特征列（交叉拟合 OOF 表, 无泄漏）
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    # 表 = ~/idsavg17/idsgnn_oof_f*.parquet（Phase 1 每折一个；缺表必须炸, 不能静默退化成 v2nowave）
    _T="${IDS_GNN_TABLE:-$(ls -1 "$HOME"/idsavg17/idsgnn_oof_f*.parquet 2>/dev/null | paste -sd,)}"
    if [ -z "$_T" ]; then
      echo "ERROR: 未找到 ids GNN OOF 表 (~/idsavg17/idsgnn_oof_f*.parquet)。先跑 Phase 1, 或 IDS_GNN_TABLE=a.parquet,b.parquet bash setup_exp.sh $V"
      exit 1
    fi
    export IDS_GNN_TABLE="$_T"
    echo "IDS_GNN_TABLE=$IDS_GNN_TABLE"
    _S=$(echo "${V#v2nowavegnn}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2nowavegs[0-9]*)  # 17.4.4 三臂实验 C 臂：nowave(纯拓扑) + GroupedBatchSampler **单独打开**、排序损失仍关
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    export USE_GROUPED_SAMPLER=1
    _S=$(echo "${V#v2nowavegs}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2wave[0-9]*)
    _S=$(echo "${V#v2wave}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2nowave[0-9]*)
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    _S=$(echo "${V#v2nowave}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2nowaver[0-9]*)  # 17.0.4 空格2: nowave(纯拓扑无近似) + 真值组内 rank 直训（§13.5 rank×最佳serve特征）
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    sed -i "s/^RANK_LOSS_W = .*/RANK_LOSS_W = 0.5/" config.py
    _S=$(echo "${V#v2nowaver}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
esac

# V3 波形消融变体（16.3.0）：v2ia<seed> 只留 ids_avg 单字段；v2cov25<seed> 25% 行覆盖率（覆盖率种子固定 42）
# v2iaa<seed>（16.10.0）：ids_avg 单字段 + 拟合回归近似（零仿真）
# 用法：v2ia42 / v2cov2542 / v2iaa42 等，后缀即 TRAIN_SEED；wave 默认开；v2iaa42m4 尾部 m4 标记
case "$V" in
  v2ia[0-9]*)
    export WAVE_FIELDS='ids_avg'
    _S=$(echo "${V#v2ia}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2iaa[0-9]*)
    export WAVE_FIELDS='ids_avg'
    export USE_IDS_AVG_APPROX=1
    _S=$(echo "${V#v2iaa}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2iag[0-9]*)    # 16.11.4: GBDT15 近似 ids_avg（USE_IDS_AVG_APPROX=2）
    export WAVE_FIELDS='ids_avg'
    export USE_IDS_AVG_APPROX=2
    _S=$(echo "${V#v2iag}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2iagr[0-9]*)   # 17.0.4 空格1: iag(GBDT15近似) + 真值组内 rank 直训（§13.5 rank×可serve特征）
    export WAVE_FIELDS='ids_avg'
    export USE_IDS_AVG_APPROX=2
    sed -i "s/^RANK_LOSS_W = .*/RANK_LOSS_W = 0.5/" config.py
    _S=$(echo "${V#v2iagr}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2iaar[0-9]*)
    export WAVE_FIELDS='ids_avg'
    export USE_IDS_AVG_APPROX=1
    sed -i "s/^RANK_LOSS_W = .*/RANK_LOSS_W = 0.5/" config.py
    _S=$(echo "${V#v2iaar}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v2cov25[0-9]*)
    export WAVE_COVERAGE='0.25'
    _S=$(echo "${V#v2cov25}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
esac

# 蒸馏变体（15.2，docs/DISTILL_PLAN.md）：v2kd<teacher><mode><seed>
#   teacher: 123=wave123 | ENS=wave42+123 平均；mode: reg(纯软标签) | rr(reg+rank)
# 例: v2kd123reg42 / v2kd123rr123 / v2kdENSrr42
case "$V" in
  v2kdwave42iaa[0-9]*)  # iaa 学生 + wave42m4 教师蒸馏(reg+rank), 16.11.32
    export WAVE_FIELDS='ids_avg'
    export USE_IDS_AVG_APPROX=1
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    sed -i "s/^KD_ENABLED = .*/KD_ENABLED = True/" config.py
    sed -i "s/^KD_MODE = .*/KD_MODE = 'reg+rank'/" config.py
    export KD_TEACHER_DIR="$HOME/project-107-v2wave42m4/outputs"
    _S=$(echo "${V#v2kdwave42iaa}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
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

# 旧三批显式钉住（18.1.0，2026-09-20）—— 因为 config.py 的 DATA_BATCHES 默认已改成 v3_delivery。
# 不钉住的话：所有既有变体名（v2 家族 + V2 时代的消融名）会**静默改吃 V3 数据，名字却一个字母没变**
# —— 历史可比性被无声打断，而且跑得完、退出码 0、日志正常（正是本仓库最怕的那类静默失败）。
# 最要命的是 v2wave42m4：DATA_SPEC_V2 §四「量化验收 B 段」把它当**跨分布通用闸门**，
# 它若变成 V3 数据，闸门就自己验自己了。
# ⚠ 只钉**名字宣称旧语义**的那些；v3* 由下面的块接管。用 ${DATA_BATCHES:-…} 保留外部显式覆盖。
# ⚠ 下列 6 个 diag 脚本各自硬编码了旧默认串（不经 config.py），**故意不动**：
#   _fit_idsavg_gnn_server.py / check_idsgnn_serve_parity.py / _serve_input_ablation.py /
#   _auto_launch_chain2.py / _auto_launch_m4_chain.py / _t_rankpairs.py
#   理由：idsavg GNN 的 OOF 表是**线上 serve 特征的来源**，给它换数据会废掉已部署特征。
#   ⇒ 它们的取数仍走旧三批，与 config.DATA_BATCHES 不再一致（这是刻意的，不是遗漏）。
case "$V" in
  v3wave[0-9]*|v3nowave[0-9]*) ;;          # V3 臂在下一块处理
  v2*|seed[0-9]*|struct*|waverich|cornerattn)
    export DATA_BATCHES="${DATA_BATCHES:-batch_v2_full,batch_v2_rest,batch_v2_m4}"
    echo "DATA_BATCHES=$DATA_BATCHES（钉住旧三批）"
    ;;
esac

# V3 单一数据集变体（18.0.0，2026-09-20）：data/v3_delivery
#   形态 = 单 circuit_static.parquet + 31 个 timing_arcs_partNN.parquet（599,976 行 / 12,455 电路）
# 用法：v3wave<seed> / v3nowave<seed>（后缀即 TRAIN_SEED；nowave = 纯拓扑，Rust 可部署形态）
# ⚠ 必须用**新名**（v3*）：:28 会 rm -rf "$D"，复用 v2 名会删掉已有 V2 运行目录。
# ⚠ v2wave[0-9]* / v2nowave[0-9]* 只匹配字面 "v2" —— 名叫 v3nowave42 **不会**命中 nowave 分支，
#   所以 v3 必须显式写臂，不能靠名字「看起来像」。
case "$V" in
  v3wave[0-9]*|v3nowave[0-9]*)
    export DATA_BATCHES=v3_delivery
    echo "DATA_BATCHES=$DATA_BATCHES"
    ;;
esac
case "$V" in
  v3wave[0-9]*)
    _S=$(echo "${V#v3wave}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v3nowave[0-9]*)
    sed -i "s/^USE_TRANSISTOR_WAVE = .*/USE_TRANSISTOR_WAVE = False/" config.py
    _S=$(echo "${V#v3nowave}" | grep -oE '^[0-9]+')
    sed -i "s/^TRAIN_SEED = .*/TRAIN_SEED = ${_S}/" config.py ;;
  v3*)
    echo "ERROR: 未知 v3 变体: $V（应为 v3wave<seed> | v3nowave<seed>）"; exit 1 ;;
esac

sed -i "s/CACHE_DIR = .*/CACHE_DIR = \"cache107$V\"/" config.py

# 可选：复用缓存（16.3.1 修正：缓存键含数据文件 mtime，必须连数据一起保 mtime 复制；16.6.0 支持 master 缓存目录）
# CACHE_SEED 支持三种源：
#   a) 旧 run 目录（如 ~/project-107-v3wave42）：含 data/ + cachev3*/cache107*
#   b) master 缓存目录（如 ~/cache107_master）：顶层直接含 graphs/ gate/ outlier/，数据退化用 ~/-project/data
# 数据从 CACHE_SEED/data 复制（cp -a 保 mtime），否则用 ~/-project/data（统一数据源，键对齐的前提）
# 例: CACHE_SEED=$HOME/cache107_master bash setup_exp.sh v2wave42
if [ -n "$CACHE_SEED" ] && [ -d "$CACHE_SEED" ]; then
  # 1) 数据（保 mtime）：fresh 总是复制（clone 的 checkout mtime 不对，必须覆盖）；
  #    RESUME 且目录已有本次所需数据 → 跳过（保留目录自身 mtime，避免图缓存键失效）
  # 18.2.0：播种清单改为**由 DATA_BATCHES 派生**，不再写死四个 batch_v2_*。
  #   为什么必须改：图缓存键含数据文件 mtime（src/train_sweep.py:291 / src/train_lib.py:245）。
  #   旧清单里没有 v3_delivery ⇒ V3 树里的 v3_delivery 只能来自 clone checkout（mtime = clone 时刻），
  #   与种子缓存建立时的 mtime 不同 ⇒ **缓存键整批落空**，从第二个 V3 运行起每趟白建一次全量图缓存
  #   （而 V3 的图比 V2 大一个量级）。V2 时代没这问题，只因那时还没有 v3_delivery 这个集。
  #   派生后与 V2 时代逻辑完全一致：V2 变体派生出旧四批（顺序不同，cp 无所谓），V3 派生出 v3_delivery。
  _SEED_MAIN="$(echo "$DATA_BATCHES" | tr ',' ' ')"
  _SEED_ALL="$_SEED_MAIN batch_v2_io"   # batch_v2_io 不在 DATA_BATCHES 里，但历史上一直在播种清单中
  _lack=""
  for b in $_SEED_ALL; do [ -d "$D/data/$b" ] || _lack="$_lack $b"; done
  if [ "$RESUME_MODE" = "1" ] && [ -z "$_lack" ]; then
    echo "RESUME: 目录已有本次所需数据（$DATA_BATCHES），跳过数据种子（保留原 mtime，缓存键不变）"
  else
    SEED_DATA="$CACHE_SEED/data"
    for b in $_SEED_MAIN; do
      if [ ! -d "$SEED_DATA/$b" ]; then
        echo "种子源 $SEED_DATA 缺 $b → 退化到统一数据源 $HOME/-project/data"
        SEED_DATA="$HOME/-project/data"
        break
      fi
    done
    _nseed=0
    for b in $_SEED_ALL; do
      if [ -d "$SEED_DATA/$b" ]; then
        mkdir -p "$D/data/$b"
        cp -a "$SEED_DATA/$b/." "$D/data/$b/"
        _nseed=$((_nseed + 1))
      fi
    done
    echo "seeded $_nseed dataset(s) (mtime preserved) from $SEED_DATA"
    for b in $_SEED_MAIN; do
      [ -d "$D/data/$b" ] || echo "⚠ WARN: $D/data/$b 缺失但本次运行需要它（DATA_BATCHES=$DATA_BATCHES）—— 图缓存键将落空"
    done
  fi
  # 2) 缓存目录：RESUME 且目录已有 graphs/ → 跳过（用自己的，增量续建）；否则复制种子
  OLD_CACHE=""
  if [ "$RESUME_MODE" = "1" ] && [ -d "$D/cache107$V/graphs" ]; then
    echo "RESUME: 目录已有图缓存，跳过缓存种子（增量续建）"
  else
    if [ -d "$CACHE_SEED/graphs" ]; then
      OLD_CACHE="$CACHE_SEED"
    else
      OLD_CACHE=$(ls -d "$CACHE_SEED"/cachev3* "$CACHE_SEED"/cache107* 2>/dev/null | head -1)
    fi
    if [ -n "$OLD_CACHE" ] && [ -d "$OLD_CACHE" ]; then
      mkdir -p "$D/cache107$V"
      cp -a "$OLD_CACHE/." "$D/cache107$V/"
      echo "seeded cache: $OLD_CACHE -> $D/cache107$V"
    fi
  fi
fi

# V3 宏表合并（**不是覆盖**）—— 两张表的 cell 名域近乎互斥（DIFF §17.2(b) 实测）：
#   共享表 24,625 条：对 V2 名域 100.0%，对 V3 名域只有 8.3%；
#   V3 自带表 1,049 条：对 V3 名域 100.0%，对 V2 名域只有 0.3%。
# 只覆盖 ⇒ V2 数据加载被打崩；只保留 ⇒ V3 有 91.7% 的 SC_ 宏落进 gate_struct 兜底
# （logic='COMPLEX' / n_t=6.0），STRUCT_MODE='base' 的 n_transistors 近乎常数。
# ⇒ 取并集：union 25,587 条，两个名域都 100%（87 个同名条目内容完全一致，合并无歧义）。
# 放在播种块**之后**：播种可能覆盖 data/sc_expansion.json；合并是 dict 更新，幂等，重复跑安全。
case "$V" in
  v3wave[0-9]*|v3nowave[0-9]*)
    ~/venv/bin/python3 scripts/merge_sc_expansion.py \
      --base data/sc_expansion.json \
      --add  data/v3_delivery/sc_expansion.json --inplace || {
        echo "ERROR: sc_expansion 合并失败 —— 拒绝以 8.3% 覆盖率的宏表启动训练"; exit 1; }
    ;;
esac

# P1-1 启动前自检（V3_ISSUES §4 第 3 条）：`transistor_count` 是否与 netlist 自洽。
#   该列同时是输入特征（data_loader.py:322）与 struct_prior[0]（:716），且 struct_encoder
#   无归一化（model.py:50-54）⇒ 同 netlist 多值 = 往模型里注入「同结构同延迟、尺寸不同」的噪声。
#   实测：V3 371/5900 种 netlist 不自洽、涉及 1,847/12,455 电路 = 14.8%，极差中位 6 / max 32；
#   而 V2 两批（batch_v2_full / batch_v2_m4）**全部自洽** ⇒ 这是 **V3 引入的回归**，不是老账。
#   归因（`scripts/diag/_t_v3_tc_cause.py`）：368/371 组「组内延迟逐值全同却多值」= 错在这一列；
#   3/371 组（20 电路）「静态六列全同、延迟不同」= 交付记录缺维度（模型侧同输入双标签）。
# ⚠ **只告警、不阻断**（脚本默认退出 0）：命中是**已记档**的缺陷，写成硬失败会把每一次 V3
#   运行都挡在门外，而它挡下的是已经知道的事、不是新信息。本检查的价值在「下一版别再犯」
#   与「将来某次运行突然变好/变坏要看得见」。生成方给出一致定义后，它应当恒静默。
case "$V" in
  v3wave[0-9]*|v3nowave[0-9]*)
    ~/venv/bin/python3 scripts/check_transistor_count.py --batches "$DATA_BATCHES" || true
    ;;
esac

ulimit -n 8192
# 蒸馏变体默认 teacher 预测目录（可被 KD_TEACHER_DIR 环境变量覆盖）
if [[ "$V" == v2kd* ]]; then
  export KD_TEACHER_DIR="${KD_TEACHER_DIR:-$HOME/project-107-v2wave123/outputs}"
fi
if [ "$RESUME_MODE" = "1" ]; then
  # RESUME 追加日志（保留上次失败/中断的现场）
  OMP_NUM_THREADS=6 nohup ~/venv/bin/python3 -u main.py >> "train107$V.log" 2>&1 &
else
  OMP_NUM_THREADS=6 nohup ~/venv/bin/python3 -u main.py > "train107$V.log" 2>&1 &
fi
echo "launched 107-$V  pid=$!  dir=$D  (RESUME=$RESUME_MODE)"
