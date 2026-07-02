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
HIDDEN_DIM = 512
NUM_LAYERS = 4
GATE_EMBED_DIM = 32   # 门类型 Embedding 维度（替代 one-hot，大幅减少参数量）
DROPOUT = 0.5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
EPOCHS = 1200
PATIENCE = 40

# 其他
RANDOM_SEED = 42

HUBER_DELTA = 0.3   # 可调整，建议从 0.2 开始尝试

# 学习率调度器配置
LR_SCHEDULER = 'ReduceLROnPlateau'     # 小数据集用 plateau，避免余弦周期震荡
LR_T_MAX = 50                          # 半个周期长度（epoch数）
LR_ETA_MIN = 1e-6                      # 最小学习率
LR_FACTOR = 0.7                        # 每次降低的倍数
LR_PATIENCE = 15                       # 验证损失连续多少epoch不下降时降低学习率
LR_MIN = 5e-6                          # 学习率下限
LR_COOLDOWN = 5                        # 降低后等待几个epoch再重新检测

# 离群点清洗（降低剔除比例）
OUTLIER_CLEANING = True
OUTLIER_TOP_PERCENT = 2
BASE_EPOCHS = 20           # 最大epoch数（早停会提前结束）
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
