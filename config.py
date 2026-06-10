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
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
EPOCHS = 500
PATIENCE = 30

# 其他
RANDOM_SEED = 42