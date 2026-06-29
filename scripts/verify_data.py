import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import glob
import pandas as pd
import torch
from src.data_loader import DelayDataset
from src.utils import split_by_circuit

def verify():
    # 只使用 batch01 数据
    static_parquets = glob.glob("data/batch_02_converted/circuit_static.parquet")
    dynamic_parquets = glob.glob("data/batch_02_converted/timing_arcs.parquet")
    print(f"Static files: {static_parquets}")
    print(f"Dynamic files: {dynamic_parquets}")

    dynamic_dfs = [pd.read_parquet(p) for p in dynamic_parquets]
    dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)
    print(f"Total rows in dynamic data: {len(dynamic_df)}")
    print(f"Columns: {dynamic_df.columns.tolist()}")
    print(f"Any null in circuit_id? {dynamic_df['circuit_id'].isna().any()}")
    print(f"Any null in DELAY? {dynamic_df['DELAY'].isna().any()}")
    print(dynamic_df.head(2))

    circuit_ids = dynamic_df['circuit_id'].unique().tolist()
    train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)
    print(f"Train circuits: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    dataset = DelayDataset(
        static_parquets=static_parquets,
        dynamic_parquets=dynamic_parquets,
        circuit_ids=train_ids,
        scaler=None,
        cache_dir=CACHE_DIR
    )
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return
    sample = dataset[0]
    print(f"Sample x shape: {sample.x.shape}")
    print(f"Sample edge_index shape: {sample.edge_index.shape}")
    print(f"Sample y: {sample.y.item():.3e}")
    has_nan = torch.isnan(sample.x).any().item()
    has_inf = torch.isinf(sample.x).any().item()
    print(f"Sample x has NaN: {has_nan}, has Inf: {has_inf}")
    if has_nan or has_inf:
        print("ERROR: Invalid values in features!")

    # 获取电路ID用于检查图构建
    row = dataset.dynamic_df.iloc[0]
    cid = row['circuit_id']
    node_names, _, _ = dataset._get_static(cid)
    print(f"Node names: {node_names[:5]}... (total {len(node_names)})")
    print("Dynamic features for first 3 nodes (last 6 columns):")
    print(sample.x[:3, -6:])

    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in loader:
        print(f"Batch x shape: {batch.x.shape}")
        print(f"Batch edge_index shape: {batch.edge_index.shape}")
        print(f"Batch y: {batch.y}")
        break
    print("Verification completed successfully.")

if __name__ == "__main__":
    verify()