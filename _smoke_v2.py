"""V2 适配冒烟：batch_v2_full(4-pin) + batch_v2_io(任意I/O) 建图/特征/前向"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.USE_V2 = True
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.data_loader import DelayDataset
from src.model import DelayGNN
from src.graph_builder import rebuild_gate_types

for batch in ['batch_v2_full', 'batch_v2_io']:
    sp = f'data/{batch}/circuit_static.parquet'
    dp = f'data/{batch}/timing_arcs.parquet'
    st = pd.read_parquet(sp)
    cells = set()
    for c in st['cell_types_json']:
        v = json.loads(c) if isinstance(c, str) else c
        if isinstance(v, list):
            cells.update(v)
    rebuild_gate_types(cells)
    ds = DelayDataset([sp], [dp], cache_dir=f'cache_smoke_{batch}', scaler=None)
    n = min(400, len(ds))
    shapes, n_nodes, n_pins = set(), set(), set()
    for i in range(n):
        d = ds[i]
        shapes.add(tuple(d.x.shape))
        n_nodes.add(d.x.shape[0])
        if hasattr(d, 'switching_pin'):
            n_pins.add(d.switching_pin)
    print(f"=== {batch}: 总样本 {len(ds)}, 抽查 {n} ===")
    print(f"  x 形状数 {len(shapes)}: {sorted(shapes)[:3]}")
    print(f"  节点数范围: {min(n_nodes)}~{max(n_nodes)}")
    print(f"  switching_pin 样例: {sorted(n_pins)[:6]}")
    # 前向
    model = DelayGNN(in_dim=ds[0].x.shape[1], hidden_dim=32, num_layers=2, num_gate_types=13)
    loader = DataLoader([ds[i] for i in range(min(16, len(ds)))], batch_size=8)
    for d in loader:
        out, node_sl = model(d.x, d.edge_index, d.batch, d.corner_cond, d.circuit_sig,
                             getattr(d, 'struct_prior', None))
        print(f"  前向 OK: out.shape={tuple(out.shape)}, node_sl.shape={tuple(node_sl.shape)}")
        assert torch.isfinite(out).all(), 'non-finite out'
        break
    print()
print("SMOKE PASS")
