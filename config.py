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
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-4
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 50

# 其他
RANDOM_SEED = 42

HUBER_DELTA = 0.5   # 增大以减少对离群值的敏感度

# 学习率调度器配置
LR_SCHEDULER = 'ReduceLROnPlateau'     # 小数据集用 plateau，避免余弦周期震荡
LR_T_MAX = 50                          # 半个周期长度（epoch数）
LR_ETA_MIN = 1e-6                      # 最小学习率
LR_FACTOR = 0.7                        # 每次降低的倍数
LR_PATIENCE = 10                       # 验证损失连续多少epoch不下降时降低学习率
LR_MIN = 5e-6                          # 学习率下限
LR_COOLDOWN = 5                        # 降低后等待几个epoch再重新检测

# 离群点清洗（降低剔除比例）
OUTLIER_CLEANING = True
OUTLIER_TOP_PERCENT = 3
BASE_EPOCHS = 30