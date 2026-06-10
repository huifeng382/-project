# analyze_errors.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_scaler
from src.data_loader import DelayDataset
from src.model import DelayGNN
import torch
from torch_geometric.loader import DataLoader

# 配置
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
STATIC_PARQUETS = ["data/batch_01/circuit_static.parquet"]  # 根据实际修改
DYNAMIC_PARQUETS = ["data/batch_01/timing_arcs.parquet"]

# 加载测试集预测结果
data = np.load(os.path.join(OUTPUT_DIR, "test_predictions.npz"))
preds = data['preds']
targets = data['targets']

# 计算相对误差
rel_error = np.abs(preds - targets) / targets * 100

# 1. 散点图：预测值 vs 真实值
plt.figure(figsize=(8,6))
plt.scatter(targets, preds, alpha=0.5, s=10)
plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
plt.xlabel("True Delay (s)")
plt.ylabel("Predicted Delay (s)")
plt.title("Prediction vs True")
plt.savefig(os.path.join(OUTPUT_DIR, "scatter.png"))
plt.close()

# 2. 相对误差分布直方图
plt.figure(figsize=(8,6))
plt.hist(rel_error, bins=50, edgecolor='black')
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.title("Relative Error Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "error_hist.png"))
plt.close()

# 3. 按电路分组分析（需要加载测试集的数据集）
# 为了得到每个样本所属的电路 ID，我们需要重新加载测试集（但使用相同的划分）
# 这里简化：直接从 Parquet 读取测试电路的动态数据（需要事先保存 test_ids）
# 假设你之前保存了 test_ids 到文件，或者从配置中重新划分
# 作为示例，我们重新划分（注意：这必须与训练时使用的划分完全一致）
from src.utils import split_by_circuit
dynamic_df = pd.concat([pd.read_parquet(p) for p in DYNAMIC_PARQUETS], ignore_index=True)
circuit_ids = dynamic_df['circuit_id'].unique().tolist()
_, _, test_ids = split_by_circuit(circuit_ids, seed=42)  # 使用相同的 RANDOM_SEED

test_dynamic = dynamic_df[dynamic_df['circuit_id'].isin(test_ids)].reset_index(drop=True)
# 注意：这里假设数据顺序与测试时 DataLoader 遍历顺序一致（测试时未 shuffle）
# 更严谨的做法是在训练时保存每个样本对应的 circuit_id 到预测结果文件。
# 这里简单演示：直接使用 test_dynamic 中记录的 circuit_id
circuit_ids_test = test_dynamic['circuit_id'].tolist()
# 确保长度匹配
assert len(circuit_ids_test) == len(rel_error)

# 按电路计算平均相对误差
df_error = pd.DataFrame({'circuit_id': circuit_ids_test, 'rel_error': rel_error})
grouped = df_error.groupby('circuit_id')['rel_error'].agg(['mean', 'std', 'count'])
print(grouped)
grouped['mean'].plot(kind='bar', yerr=grouped['std'], capsize=2)
plt.ylabel("Mean Relative Error (%)")
plt.title("Error per Circuit")
plt.savefig(os.path.join(OUTPUT_DIR, "per_circuit_error.png"))
plt.close()

# 4. 按切换引脚分组分析
switching_pins = test_dynamic['switching_pin'].tolist()
df_error_sw = pd.DataFrame({'pin': switching_pins, 'rel_error': rel_error})
pin_stats = df_error_sw.groupby('pin')['rel_error'].mean()
print("Error by switching pin:\n", pin_stats)

print("Analysis completed. Figures saved in", OUTPUT_DIR)