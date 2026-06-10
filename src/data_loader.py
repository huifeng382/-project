import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from src.graph_builder import build_static_graph, GATE_TYPES
from src.utils import load_scaler

class DelayDataset(Dataset):
    def __init__(self, static_parquets, dynamic_parquets, circuit_ids=None, scaler=None, cache_dir="cache"):
        # 统一转换为列表
        if isinstance(static_parquets, str):
            static_parquets = [static_parquets]
        if isinstance(dynamic_parquets, str):
            dynamic_parquets = [dynamic_parquets]

        # 读取并合并静态数据
        static_dfs = []
        for p in static_parquets:
            df = pd.read_parquet(p)
            static_dfs.append(df)
        self.static_df = pd.concat(static_dfs).drop_duplicates('circuit_id').set_index('circuit_id')

        # 读取并合并动态数据
        dynamic_dfs = []
        for p in dynamic_parquets:
            df = pd.read_parquet(p)
            dynamic_dfs.append(df)
        self.dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)

        # 筛选电路（如果指定）
        if circuit_ids is not None:
            self.dynamic_df = self.dynamic_df[self.dynamic_df['circuit_id'].isin(circuit_ids)].reset_index(drop=True)

        # 确保 vector 列格式正确
        self.dynamic_df['vector'] = self.dynamic_df['vector'].astype(str).str.zfill(5)

        self.scaler = scaler
        self.cache_dir = cache_dir
        self.graph_cache = {}
        self._prepare_static_graphs()

    def _prepare_static_graphs(self):
        for cid in self.dynamic_df['circuit_id'].unique():
            netlist = self.static_df.loc[cid, 'gate_level_netlist']
            node_names, node_static, edge_index = build_static_graph(cid, netlist)
            self.graph_cache[cid] = (node_names, node_static, edge_index)   # 存储 node_type_enc
    def _get_static(self, cid):
        return self.graph_cache[cid]

    def _get_dynamic_features(self, row):
        pins = ['a', 'b', 'c', 'd', 'e']
        vector = row['vector']
        logic = {pin: int(vector[i]) for i, pin in enumerate(pins)}
        switching = row['switching_pin']
        dyn_feats = {}
        for pin in pins:
            feat = [
                float(logic[pin]),
                1.0 if pin == switching else 0.0,
                row[f'slew_{pin}'],
                row[f'arrival_{pin}'],
                row[f'load_{pin}'],
                0.0  # output_load placeholder
            ]
            if self.scaler is not None:
                continuous = np.array([feat[2], feat[3], feat[4]]).reshape(1, -1)
                scaled_cont = self.scaler.transform(continuous)[0]
                feat[2], feat[3], feat[4] = scaled_cont[0], scaled_cont[1], scaled_cont[2]
            dyn_feats[pin] = feat
        return dyn_feats

    def __len__(self):
        return len(self.dynamic_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        else:
            idx = int(idx)
        row = self.dynamic_df.iloc[idx]
        cid = row['circuit_id']
        node_names, node_static, edge_index = self._get_static(cid)
        dyn_feats = self._get_dynamic_features(row)

        num_nodes = len(node_names)
        node_feat_dim = node_static.shape[1] + 6
        x = torch.zeros((num_nodes, node_feat_dim), dtype=torch.float)
        x[:, :node_static.shape[1]] = node_static

        for i, n in enumerate(node_names):
            if n in dyn_feats:
                dyn = dyn_feats[n]
                x[i, -6:] = torch.tensor(dyn, dtype=torch.float)

        y = torch.tensor([row['DELAY']], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        # 不再设置 data.switching_pin
        data.switching_pin = row['switching_pin']
        return data