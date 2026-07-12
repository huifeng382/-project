import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from config import *
from src.data_loader import DelayDataset
from src.model import DelayGNN
from src.train_sweep import evaluate  # 复用评估函数
from src.utils import set_seed, create_dir

def ensemble_predict(models, data_loader, device):
    """对每个样本，取所有模型预测的均值"""
    all_preds = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            preds = []
            for model in models:
                model.eval()
                out = model(data.x, data.edge_index, data.batch)[0]
                preds.append(out.cpu().numpy())
            avg_pred = np.mean(preds, axis=0)  # shape: (batch_size,)
            all_preds.append(avg_pred)
    return np.concatenate(all_preds)

def main():
    # 设置种子（不重要，但保持一致性）
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义要集成的种子列表
    seeds = [42, 123, 456, 789, 1010]
    models = []
    for seed in seeds:
        model_path = f"outputs/seed_{seed}/best_model.pt"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping.")
            continue
        # 需要与训练时相同的 in_dim, hidden_dim, num_layers, dropout
        # 这里从数据集中获取 in_dim，或者先加载一个样本确定
        # 由于我们不知道 in_dim，可以先用一个 dummy 数据集获取
        # 但更方便的是，从已经训练好的模型状态中恢复结构（但状态不包含结构）
        # 所以我们仍需提前知道 in_dim，这里假设从 config 或从数据加载
        # 方案：加载一个测试数据集样本获取 in_dim
        # 注意：这里需要与训练时数据预处理完全一致（scaler, 文件路径等）
        static_parquets = glob.glob("data/batch_*/circuit_static.parquet")
        dynamic_parquets = glob.glob("data/batch_*/timing_arcs.parquet")
        # 构建一个临时的数据集只是为了获取特征维度，但这里简单起见直接硬编码或从 config 读
        # 我们可以从 config 的 HIDDEN_DIM 等获取，但 in_dim 由数据决定
        # 更好的方法是保存一个元数据文件，但这里我们临时取一个样本
        # 先构建测试数据集
        from src.data_loader import DelayDataset
        # 由于我们需要 scaler，但测试时也需要 scaler，可以从训练输出加载 scaler.pkl
        from src.utils import load_scaler
        scaler = load_scaler("outputs/seed_42/scaler.pkl")  # 使用第一个种子的 scaler（应该一样）
        test_dataset = DelayDataset(static_parquets, dynamic_parquets,
                                    circuit_ids=None, scaler=scaler, cache_dir=CACHE_DIR)
        # 取第一个样本获取 in_dim
        sample = test_dataset[0]
        in_dim = sample.x.shape[1]
        # 现在构建模型
        model = DelayGNN(in_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        models.append(model)
        print(f"Loaded model from seed {seed}")

    if not models:
        raise RuntimeError("No models loaded.")

    # 构建测试数据加载器（与训练时保持一致）
    # 注意：测试集划分可能与训练时不同（使用了不同的种子划分），但为了公平，我们在训练时已经固定了划分种子（RANDOM_SEED）
    # 但训练时的划分是用全局种子，现在集成时我们需要确保使用相同的划分，即训练脚本中的 RANDOM_SEED=42 进行划分。
    # 实际上，划分是在 train.py 中根据 RANDOM_SEED（未修改）进行的，所以我们不需要额外处理，直接使用测试集。
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 集成预测
    preds = ensemble_predict(models, test_loader, device)

    # 获取真实标签（从测试集）
    targets = []
    for data in test_loader:
        targets.append(data.y.cpu().numpy())
    targets = np.concatenate(targets)

    # 计算相对误差
    rel_error = np.abs(preds - targets) / targets * 100
    mean_rel_error = np.mean(rel_error)
    print(f"Ensemble Test Mean Relative Error: {mean_rel_error:.2f}%")

    # 保存预测结果
    np.savez("outputs/ensemble_predictions.npz", preds=preds, targets=targets)

if __name__ == "__main__":
    main()