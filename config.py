# config.py
import os

# 路径配置
DATA_DIR = "data"
CIRCUIT_CSV = os.path.join(DATA_DIR, "circuit_dataset.csv")
STATIC_JSON = os.path.join(DATA_DIR, "static_features.json")
CELL_LUT_JSON = os.path.join(DATA_DIR, "std_cells_lut.json")
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"

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
TRAIN_SEED = 42            # 只控制模型初始化+训练shuffle（变它→不同模型，同切分）

# best_model 选点指标：val_rel_err / val_loss / smoothed_rel_err(滑动平均去噪)
# 12.x 实测 smoothed_rel_err(bmsm) 排序全指标最好、无短板 → 设为默认
BEST_MODEL_METRIC = 'smoothed_rel_err'
BEST_SMOOTH_WINDOW = 5     # smoothed_rel_err 的滑动窗口

# 组内成对排序损失（直接优化「分辨同组变体谁更快」，尤其小幅差异）
RANK_LOSS_W = 0.0          # 0=关(默认,不改现有行为)；>0 启用，用 GroupedBatchSampler
RANK_MARGIN = 0.03         # log10 延迟空间的间隔（≈7% 相对）

# 按排序指标选 checkpoint（直接对齐变体择优任务，替换 smoothed_rel_err 选点）
BEST_RANK_METRIC = 'none'  # 'none'(沿原行为) | 'regret'(选val选择遗憾最小) | 'spearman'(选val秩相关最高)
RANK_EVAL_INTERVAL = 5     # 每隔 N 个 epoch 在 val 上评估排序

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
PLATEAU_MIN_EPOCHS = 50    # 最少训练epoch，在此之前不触发平台早停

# 测试模式：快速检测平台期，大幅缩短训练时间
# 开启后 err 不再明显下降即自动停止，关闭后可获得更精确的最优 err
QUICK_TEST = False         # True=测试模式（提前停止），False=正常模式
QUICK_MIN_EPOCHS = 30      # 最少训练epoch
QUICK_WINDOW = 20          # 观察窗口
QUICK_MIN_DELTA = 1.0      # best err 在窗口内至少下降这么多百分点，否则停止

# 数据筛选：只保留标准4引脚（a,b,c,d）电路，去除图结构不一致的电路
FOUR_PIN_ONLY = True       # True=只保留4引脚电路（去掉~12%），False=保留全部

# 新物理特征开关（delivery1 数据已提供，独立控制，默认全关=纯基线）
USE_PARASITIC_CAPS = False     # 每门寄生电容 -> 1 个节点特征
USE_TRANSISTOR_WAVE = False    # 晶体管波形 -> 3 个节点特征(ids_avg/ids_peak/vds_swing 按门聚合均值)
USE_SUPPLY_NOISE = False       # 电源噪声 -> 2 个节点特征(vdd_droop_mV/gnd_bounce_mV, 广播到所有节点)

# LIB (Scheme A) 损失权重：SC 宏展开→标准单元链查表
LIB_AUX_W = 0.1            # LIB 辅助损失总权重（外层缩放）
LIB_TOTAL_W = 0.1         # 总延迟（展开求和 vs 实测 DELAY）项
PG_DELAY_W = 0.5          # 逐门 delay 监督（per_gate_delay, 60%）
PG_OUTSLEW_W = 0.1        # 逐门输出 slew 监督（per_gate_out_slew, 100%）
PG_INSLEW_W = 0.1         # 逐门输入 slew 监督（per_gate_in_slew, 60%）
