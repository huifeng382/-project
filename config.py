# config.py
import os

# 路径配置
DATA_DIR = "data"
CIRCUIT_CSV = os.path.join(DATA_DIR, "circuit_dataset.csv")
STATIC_JSON = os.path.join(DATA_DIR, "static_features.json")
CELL_LUT_JSON = os.path.join(DATA_DIR, "std_cells_lut.json")
CACHE_DIR = os.environ.get('CACHE_DIR', 'cache')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')

# 模型超参数
HIDDEN_DIM = 256
NUM_LAYERS = 6
GATE_EMBED_DIM = 32   # 门类型 Embedding 维度（替代 one-hot，大幅减少参数量）
DROPOUT = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 80
EPOCHS = 1200
PATIENCE = 40

# 其他
RANDOM_SEED = 42

# 种子解耦：切分与训练分开，便于「同一切分、不同初始化」做集成/方差
SPLIT_SEED = 42            # 只控制 train/val/test 切分（固定→测试集不变，可集成）
TRAIN_SEED = int(os.environ.get('TRAIN_SEED', '42'))   # 只控制模型初始化+训练shuffle（变它→不同模型，同切分）

# best_model 选点指标：capture2 / smoothed_rel_err / val_loss / val_rel_err
# 12.x 实测 smoothed_rel_err(bmsm) 排序全指标最好、无短板 → 曾为默认
# 17.3.8：改用 capture2（两阶段捕获率，val 上取最大）。理由：它直接对应部署口径
#   「GNN 出前3 → SPICE 精排 → 取前3内真最优」，且分母是「可改进空间」→ 随 spread 归一、
#   跨组可比。smoothed_rel_err 属回归误差型（预测准不准），而部署只消费组内变体次序，
#   两者不是一回事（实测该 score/smoothed 序与 Rust 部署序可反序）。
# ⚠ 2026-09-17 追记（**判决未落地，替换待定 —— 见 task #60 D**）：capture2 后来被**判 ❌ 不采纳**
#   （四条证据：过度遗憾翻倍 / 亚噪声信号 0.16~0.24pp 撬动 1~2.65pp 后果 / Spearman-Pearson 变号 /
#   跨臂符号反转），但这一行仍是 capture2 → 「判了没改」。**本项只决定 best_model.pt 存哪个 epoch**，
#   不碰梯度/RNG（改它不影响训练本身）。而按 OPERATIONS §6.7:212，best_model.pt **已不进部署候选**
#   → 对部署无害，故不擅自替换。候选：'smoothed_rel_err'（config 上方记的 12.x 实测全指标无短板、
#   且曾是默认，窗口 BEST_SMOOTH_WINDOW 仍在）/ 'val_loss' / 'val_rel_err'。**待用户定档后再改。**
BEST_MODEL_METRIC = 'capture2'
BEST_SMOOTH_WINDOW = 5     # smoothed_rel_err 的滑动窗口

# 组内成对排序损失（直接优化「分辨同组变体谁更快」，尤其小幅差异）
RANK_LOSS_W = 0.0          # 0=关(默认,不改现有行为)；>0 启用。**采样器不再隐含在这里** → USE_GROUPED_SAMPLER
RANK_MARGIN = 0.03         # log10 延迟空间的间隔（≈7% 相对）

# 17.4.3 采样器与排序损失**解耦**。原先 `RANK_LOSS_W > 0` 一处开关同时改了「损失项」和
#   「采样器」，于是 §13.6「排序损失全轴净伤害」里这**两个效应本来就分不开**。
# ⚠ 为何必须能单独打开采样器：本地核算实测，**默认路径**的原 sampler（站点1 else =
#   `CircuitGroupSampler` 整电路打包）下一个 80 行的 batch **零成对样本的占 full 92.1% /
#   rest 49.0% / m4 26.7%**（Grouped 下为 0.0%，成对均值比 102~812×）—— 组均 11.8 行，
#   整电路打包仍常把一组拆到多个 batch ⇒ 成对项**大量为空**，不是「效果差」。
#   故 GroupedBatchSampler 不是「附带的混淆」，而是成对项非空的**前提**。
#   ⚠ 17.4.4 更正：本行原记「原 sampler（随机 shuffle）rest 94.0 / m4 67.3 / full 54.2」是错的
#   —— 那三个数是**随机置换**的读数，而随机置换只对应站点2（离群点清洗分支）的 else，
#   不是默认路径。方向不变（Grouped 仍是前提），但三个数须换成上面那组。
#   真读数见 scripts/diag/_t_sampler_live.py Part C（exec 站点真源码 + 真 CircuitGroupSampler）。
#   'auto' = 旧行为（Grouped ⟺ RANK_LOSS_W > 0），**默认，逐字不改现有一切 run**。
#   '1'/'0'（及 true/false/yes/no/on/off）= 显式指定，与 RANK_LOSS_W 无关。
#   三臂实验（task #60）：A = auto 且 w=0（基线）/ C = USE_GROUPED_SAMPLER=1 且 w=0（新增，
#   单独隔离采样器效应）/ B = 1 且 w>0（= 已跑过的 v2nowaver42m4）⇒ C−A = 纯采样器、B−C = 纯损失。
#   可用环境变量覆盖以便不改文件：USE_GROUPED_SAMPLER=1 python3 main.py
USE_GROUPED_SAMPLER = os.environ.get('USE_GROUPED_SAMPLER', 'auto')

# 按排序指标选 checkpoint（直接对齐变体择优任务，替换 smoothed_rel_err 选点）
BEST_RANK_METRIC = 'none'  # 'none'(沿原行为) | 'regret'(选val选择遗憾最小) | 'spearman'(选val秩相关最高)
RANK_EVAL_INTERVAL = 5     # 每隔 N 个 epoch 在 val 上评估排序

# 17.4.1 部署选点政策落地（OPERATIONS §6.7:212）—— midpoint 选点规则。
#   政策原文：「取平台末端（最后一个 midpoint），不取 shadow argmax；best_model.pt 不进部署候选」。
#   'last'            = 取**平台末端** = epoch 最大的有效 midpoint。**政策默认**，确定性、不依赖 shadow。
#   'argmax_capture2' = 17.3.7~17.4.0 旧行为（val capture2 取 argmax）。**保留仅供复现旧 run**：
#                       在不可分辨的差上取 argmax = 拟合噪声（已证严格@3 对 1 ulp 的敏感度是遗憾的 ~17 倍），
#                       且该 score 序与 Rust 部署序实测 Spearman = −1（42b 五点完全反序）。
#   ⚠ 改这一项**不影响训练**（不碰梯度/RNG），只改变「哪个 ckpt 被当成产物」。两个调用点
#   （训练内 midpoint 块 / EVAL_ONLY=midpoint）共用 src.train_sweep.pick_midpoint，不会各走各的。
#   可用环境变量覆盖以便不改文件复现旧 run：MIDPOINT_SELECT=argmax_capture2 python3 main.py
MIDPOINT_SELECT = os.environ.get('MIDPOINT_SELECT', 'last')

# 17.3.8 早停守卫：val_loss 平台先到、而 capture2 仍在刷新时，先不停（**只许延后，不许提前**）。
# 语义是「给早停**增加**一个条件」而非放宽：原 val_loss 条件仍须成立，只是额外要求 capture2
#   也已平台。动机：早停不改变走过的路，但改变「路有多长」→ 改变候选集（midpoint_ep*.pt 的范围）。
#   若捕获率的最优点恰好落在最后一个可用中点上，旧的 val_loss 早停就是把它砍掉了。
# ⚠ 默认 False：42m4(停 ep310)/42b(停 ep269) 的**旧判据**最优点都在中点区间内部，「被截断」
#   目前没有证据。先做只读复算看清 val 上逐中点的 capture2 曲线，再决定要不要打开这个开关。
EARLYSTOP_CAP2_GUARD = False   # True = 早停须 val_loss 与 capture2 双平台
CAP2_GUARD_PATIENCE = 100      # capture2 连续多少 epoch 未刷新最大值算平台（≈2 个 midpoint 间隔）
CAP2_GUARD_MAX_EXTRA = 200     # 守卫最多额外延长多少 epoch（硬上限，防噪声微升把 run 拖到 EPOCHS）

HUBER_DELTA = 0.3   # 可调整，建议从 0.2 开始尝试

# 学习率调度器配置
LR_SCHEDULER = 'ReduceLROnPlateau'     # 小数据集用 plateau，避免余弦周期震荡
LR_T_MAX = 50                          # 半个周期长度（epoch数）
LR_ETA_MIN = 1e-6                      # 最小学习率
LR_FACTOR = 0.5                        # 每次降低的倍数（10.6 深退火：更陡）
LR_PATIENCE = 15                       # 验证损失连续多少epoch不下降时降低学习率
LR_MIN = 1e-6                          # 学习率下限（10.6 深退火：更低）
LR_COOLDOWN = 5                        # 降低后等待几个epoch再重新检测

# 离群点清洗（降低剔除比例）
OUTLIER_CLEANING = True
OUTLIER_TOP_PERCENT = 2
BASE_EPOCHS = 10           # 最大epoch数（早停会提前结束）
BASE_MIN_EPOCHS = 5        # 最少训练epoch
BASE_PATIENCE = 5          # loss连续不下降则早停
BASE_MIN_DELTA = 0.001     # 视为改进的最小loss下降

# 智能早停：检测过拟合平台期，提前终止训练
PLATEAU_WINDOW = 25        # 观察窗口（epoch数）
PLATEAU_MIN_DELTA = 0.3    # val err 至少下降这么多才算有效改善（百分点）
PLATEAU_MIN_EPOCHS = 150    # 最少训练epoch，在此之前不触发平台早停

# 测试模式：快速检测平台期，大幅缩短训练时间
# 开启后 err 不再明显下降即自动停止，关闭后可获得更精确的最优 err
QUICK_TEST = False         # True=测试模式（提前停止），False=正常模式
QUICK_MIN_EPOCHS = 30      # 最少训练epoch
QUICK_WINDOW = 20          # 观察窗口
QUICK_MIN_DELTA = 1.0      # best err 在窗口内至少下降这么多百分点，否则停止

# 数据筛选：只保留标准4引脚（a,b,c,d）电路，去除图结构不一致的电路
FOUR_PIN_ONLY = True       # True=只保留4引脚电路（去掉~12%），False=保留全部

# V2 数据模式（15.1.0 起）：True=用 batch_v2_full(4-pin) + batch_v2_io(任意I/O)，False=旧 delivery1+2
# 15.1.1 起默认 True：训练只使用 V2 两个新数据集
USE_V2 = True
# V2 训练推荐（14.4.4 结论）：STRUCT_MODE='logic_only'（干净 10 逻辑最优）；cornerattn 保留默认
V2_STRUCT_MODE = 'logic_only'

# 新物理特征开关（delivery1 数据已提供，独立控制，默认全关=纯基线）
USE_PARASITIC_CAPS = False     # 每门寄生电容 -> 1 个节点特征
USE_TRANSISTOR_WAVE = os.environ.get('USE_TRANSISTOR_WAVE', '1') == '1'   # 晶体管波形 -> 3 个节点特征（13.5 消融实验证明有效，设为默认；0=no-wave）
USE_SUPPLY_NOISE = False       # 电源噪声 -> 2 个节点特征(vdd_droop_mV/gnd_bounce_mV, 广播到所有节点)

# 17.1.2: GNN 预测的 per-gate ids_avg 特征列 —— 独立 extra_feats 块（不在 USE_TRANSISTOR_WAVE 里），
#   表 = idsavg GNN 的 OOF 推理产物（scripts/diag/_fit_idsavg_gnn_server.py 的 INFER_CKPT/PRED_OUT 模式）:
#   列 circuit_id/switching_pin/direction/output/corner/gate/pred_log1p。
#   取 expm1(pred_log1p) 喂入 → 与真实 ids_avg / 近似槽(USE_IDS_AVG_APPROX)同尺度，可直接对比。
#   查表键 = (circuit_id, switching_pin, direction, output, corner)，门名小写；缺失 → 0（同 wave 缺失行为）。
IDS_GNN_TABLE = os.environ.get('IDS_GNN_TABLE', '')   # 非空 = 表路径 → 启用该 1 列节点特征（多折可逗号分隔）

# 蒸馏（KD）：teacher 有 wave → student 无 wave（Rust 集成用，见 docs/DISTILL_PLAN.md）
# student 训练时按 dataset row_idx 索引 teacher 预测；KD_ENABLED=1 生效
KD_ENABLED = os.environ.get('KD_ENABLED', '0') == '1'          # 1=启用蒸馏损失
KD_TEACHER_DIR = os.environ.get('KD_TEACHER_DIR', '')          # kd_teacher_preds_{train,val,test}.npy 所在目录
KD_LAMBDA = float(os.environ.get('KD_LAMBDA', '1.0'))          # 软标签回归权重（MSE on log10 预测）
KD_RANK_W = float(os.environ.get('KD_RANK_W', '1.0'))          # teacher 排序监督权重（复用 _pairwise_rank_loss）
KD_MODE = os.environ.get('KD_MODE', 'reg+rank')                # 'reg' | 'rank' | 'reg+rank'
# teacher 预测导出模式（一次性，在 teacher 目录跑）：KD_PREDS_ONLY=1 KD_TEACHER_CKPT=<ckpt> KD_TEACHER_DIR=<out>
KD_PREDS_ONLY = os.environ.get('KD_PREDS_ONLY', '0') == '1'
KD_TEACHER_CKPT = os.environ.get('KD_TEACHER_CKPT', '')        # teacher checkpoint（midpoint_ep*.pt / best_model.pt）

# 结构先验特征（Task #8 分析：transistor_count + SC_AND/SC_INV_WIRE 计数 -> 图级残差）
USE_STRUCT_PRIOR = True        # 分析发现的全局结构信号，以残差形式注入 pooling 后（13.4 采纳为默认）

# 新探索：基于 delivery1+2 全量数据的进一步改进
WAVE_AGG_RICH = False          # 晶体管波形聚合: mean → (mean, max, std)（in_dim 17→23）
WAVE_FIELDS = [f.strip() for f in os.environ.get('WAVE_FIELDS', 'ids_avg,ids_peak,vds_swing').split(',') if f.strip()]
WAVE_COVERAGE = float(os.environ.get('WAVE_COVERAGE', '1.0'))   # 行覆盖率（1.0=全；<1 模拟部分仿真）
WAVE_COVERAGE_SEED = int(os.environ.get('WAVE_COVERAGE_SEED', '42'))  # 固定掩蔽种子（跨 epoch 不变）
USE_IDS_AVG_APPROX = os.environ.get('USE_IDS_AVG_APPROX', '0')  # 16.10.0: '1'=线性拟合回归近似 ids_avg（零仿真，Rust 端可算）；16.11.4: '2'=GBDT15 近似
IDS_AVG_APPROX_COEF = [0.133045, 0.083942, 0.083942, 0.078951, 1.036561, -1.564259, 0.133045, 0.083942, 0.121103]  # 来自 _eval_idsavg_approx.py（full 样本回归 R^2=0.655）
MIN_GROUP_SIZE = int(os.environ.get('MIN_GROUP_SIZE', '10'))   # 组大小过滤：剔除 <N 变体的组（排序无价值）
DATA_BATCHES = os.environ.get('DATA_BATCHES', 'batch_v2_full,batch_v2_rest,batch_v2_m4')  # 训练数据批次（16.11.4 起默认 full+rest+m4，m4=V3.2 五形状补充）
GRAPH_CACHE_MAX = int(os.environ.get('GRAPH_CACHE_MAX', '6000'))  # 图 LRU 缓存上限（内存驻留图数，超限磁盘回源；防多 run 并发内存过载）
USE_CORNER_ATTN = True         # Corner 感知注意力池化（13.6 内部最优，设为默认）

# 结构特征模式（14.2.2 起：用逻辑类别+结构特征替代 638 类 cell 名嵌入）
# 'base'       = 10逻辑 + n_transistors（主测）
# 'logic_only' = 只 10逻辑，无 n_t（消融：n_t 有无贡献）
# 'rich'       = 10逻辑 + n_t + stack + parallel（更多结构细节）
# 'elec'       = 10逻辑 + n_t + p/g/drive 改从 ASAP7 结构算（修 p/g 名字正则失效）
STRUCT_MODE = 'base'

# 中途快照：每隔 N epoch 保存 checkpoint，训练结束后自动选最优 epoch
SAVE_MIDPOINTS = True       # 是否保存中途 checkpoint（不影响训练 RNG，默认开启）
MIDPOINT_INTERVAL = 50      # 每隔多少 epoch 保存一次

# LIB (Scheme A) 损失权重：SC 宏展开→标准单元链查表
LIB_AUX_W = 0.1            # LIB 辅助损失总权重（外层缩放）
LIB_TOTAL_W = 0.1         # 总延迟（展开求和 vs 实测 DELAY）项
PG_DELAY_W = 0.5          # 逐门 delay 监督（per_gate_delay, 60%）
PG_OUTSLEW_W = 0.1        # 逐门输出 slew 监督（per_gate_out_slew, 100%）
PG_INSLEW_W = 0.1         # 逐门输入 slew 监督（per_gate_in_slew, 60%）
