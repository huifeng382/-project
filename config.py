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
NUM_LAYERS = 2
DROPOUT = 0.4
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
EPOCHS = 1200
PATIENCE = 80

# 其他
RANDOM_SEED = 42

HUBER_DELTA = 0.3   # 可调整，建议从 0.2 开始尝试

# 学习率调度器配置
LR_SCHEDULER = 'ReduceLROnPlateau'   # 可选: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR'
LR_FACTOR = 0.7                      # 每次降低的倍数
LR_PATIENCE = 10                     # 验证损失连续多少epoch不下降时降低学习率
LR_MIN = 1e-6                        # 学习率下限
LR_COOLDOWN = 0                      # 降低后等待几个epoch再重新检测