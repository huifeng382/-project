import os
import sys

# 获取当前脚本所在目录的上级目录（即项目根目录）
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)   # 将项目根目录加入模块搜索路径

import torch
import pandas as pd
from src.data_loader import DelayDataset

def test_loader():
    # 使用绝对路径读取 Parquet 文件
    static_path = os.path.join(project_root, "data", "circuit_static.parquet")
    dynamic_path = os.path.join(project_root, "data", "timing_arcs.parquet")
    
    # 检查文件是否存在
    if not os.path.exists(static_path):
        print(f"错误：找不到静态文件 {static_path}")
        return
    if not os.path.exists(dynamic_path):
        print(f"错误：找不到动态文件 {dynamic_path}")
        return
    
    dynamic_df = pd.read_parquet(dynamic_path)
    all_ids = dynamic_df['circuit_id'].unique().tolist()
    train_ids = all_ids[:3]   # 只取前3个电路快速测试
    
    dataset = DelayDataset(
        static_parquets=[static_path],
        dynamic_parquets=[dynamic_path],
        circuit_ids=train_ids,
        scaler=None,
        cache_dir=os.path.join(project_root, "cache")
    )
    
    print(f"Dataset size: {len(dataset)}")
    data = dataset[0]
    print("Sample graph:")
    print(f"  Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Delay: {data.y.item():.3e}")

if __name__ == "__main__":
    test_loader()