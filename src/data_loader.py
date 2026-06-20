import pandas as pd
import numpy as np
import torch
import json
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

        # ------------------- 列名规范化函数 -------------------
        def normalize_static(df):
            # 电路 ID
            if 'circuit_id' not in df.columns:
                if 'candidate' in df.columns:
                    df = df.rename(columns={'candidate': 'circuit_id'})
                elif 'candidate_id' in df.columns:
                    df = df.rename(columns={'candidate_id': 'circuit_id'})
                else:
                    raise KeyError(f"Static data missing id column. Columns: {df.columns.tolist()}")
            df['circuit_id'] = df['circuit_id'].astype(str)

            # 网表列
            if 'gate_level_netlist' not in df.columns:
                if 'gate_level_netlist_std' in df.columns:
                    df = df.rename(columns={'gate_level_netlist_std': 'gate_level_netlist'})
                else:
                    raise KeyError(f"Static data missing netlist column. Columns: {df.columns.tolist()}")

            # 解析 pin_loads_json
            if 'pin_loads_json' in df.columns:
                def parse_loads(loads_str):
                    try:
                        return json.loads(loads_str)
                    except:
                        return {}
                df['pin_loads_dict'] = df['pin_loads_json'].apply(parse_loads)
            else:
                df['pin_loads_dict'] = [{}] * len(df)

            # 输出负载
            if 'output_load' not in df.columns and 'output_load_f' in df.columns:
                df = df.rename(columns={'output_load_f': 'output_load'})
            return df

        def normalize_dynamic(df):
            # 电路 ID
            if 'circuit_id' not in df.columns:
                if 'candidate' in df.columns:
                    df = df.rename(columns={'candidate': 'circuit_id'})
                elif 'candidate_id' in df.columns:
                    df = df.rename(columns={'candidate_id': 'circuit_id'})
                else:
                    raise KeyError(f"Dynamic data missing id column. Columns: {df.columns.tolist()}")
            df['circuit_id'] = df['circuit_id'].astype(str)

            # 延迟列名统一为 DELAY
            if 'DELAY' not in df.columns:
                for col in ['delay', 'delay_s', 'Delay', 'delays']:
                    if col in df.columns:
                        df = df.rename(columns={col: 'DELAY'})
                        break

            # 确保 vector 列存在且格式正确
            if 'vector' in df.columns:
                df['vector'] = df['vector'].astype(str).str.zfill(5)

            return df
        # -----------------------------------------------------

        # 读取并合并静态数据
        static_dfs = []
        for p in static_parquets:
            df = pd.read_parquet(p)
            df = normalize_static(df)
            static_dfs.append(df)
        self.static_df = pd.concat(static_dfs).drop_duplicates('circuit_id').set_index('circuit_id')

        # 读取并合并动态数据
        dynamic_dfs = []
        for p in dynamic_parquets:
            df = pd.read_parquet(p)
            df = normalize_dynamic(df)
            dynamic_dfs.append(df)
        self.dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)

        # 筛选电路（如果指定）
        if circuit_ids is not None:
            self.dynamic_df = self.dynamic_df[self.dynamic_df['circuit_id'].isin(circuit_ids)].reset_index(drop=True)

        # 剔除 DELAY 为 NaN 的样本
        if 'DELAY' in self.dynamic_df.columns:
            self.dynamic_df = self.dynamic_df.dropna(subset=['DELAY']).reset_index(drop=True)

        # 确保 vector 列格式正确（再次保证）
        if 'vector' in self.dynamic_df.columns:
            self.dynamic_df['vector'] = self.dynamic_df['vector'].astype(str).str.zfill(5)

        # 从静态数据中动态推断输入引脚（替代硬编码）
        if 'input_pins_json' in self.static_df.columns:
            all_pins = set()
            for pins_json in self.static_df['input_pins_json']:
                all_pins.update(json.loads(pins_json))
            self.pins = sorted(all_pins)
        else:
            # fallback 1: 从第一个网表的 .SUBCKT DUT 行解析
            sample_netlist = self.static_df['gate_level_netlist'].iloc[0]
            self.pins = []
            for line in sample_netlist.split('\n'):
                if line.strip().upper().startswith('.SUBCKT DUT'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        self.pins = [
                            p for p in parts[2:]
                            if p.lower() not in ('vdd', 'gnd', 'vss', 'out')
                        ]
                    break

        # fallback 2: 如果网表也没有输入引脚，从动态数据的 slew_*/arrival_* 列名推断
        if not self.pins:
            pin_cols = [c for c in self.dynamic_df.columns if c.startswith('slew_')]
            candidate_pins = sorted([c[5:] for c in pin_cols])
            # 过滤：只保留在 switching_pin 中实际出现过的引脚（排除参考信号如 s）
            actual_pins = set(self.dynamic_df['switching_pin'].dropna().unique())
            self.pins = [p for p in candidate_pins if p in actual_pins]

        self.scaler = scaler
        self.cache_dir = cache_dir
        self.graph_cache = {}
        self._prepare_static_graphs()

    def _prepare_static_graphs(self):
        for cid in self.dynamic_df['circuit_id'].unique():
            netlist = self.static_df.loc[cid, 'gate_level_netlist']
            node_names, node_static, edge_index = build_static_graph(cid, netlist)
            self.graph_cache[cid] = (node_names, node_static, edge_index)

    def _get_static(self, cid):
        return self.graph_cache[cid]

    def _get_dynamic_features(self, row, pin_loads_dict):
        pins = self.pins
        switching = row['switching_pin']
        direction = row['direction']
        # 从 direction 推断 switching_pin 的切换前状态
        # rise: 0 -> 1，切换前为 0; fall: 1 -> 0，切换前为 1
        switching_before = 0.0 if direction == 'rise' else 1.0
        dyn_feats = {}
        for pin in pins:
            # 负载：优先从动态数据读取 load_{pin}，否则从静态字典
            load_col = f'load_{pin}'
            if load_col in row.index and pd.notna(row[load_col]):
                load_val = row[load_col]
            else:
                load_val = pin_loads_dict.get(pin, 0.0)

            # 获取 slew 和 arrival（支持多种列名格式）
            slew_col = f'slew_{pin}'
            if slew_col in row.index and pd.notna(row[slew_col]):
                slew_val = row[slew_col]
            else:
                slew_val = 0.0

            # arrival 列可能是 arrival_{pin} 或 arrival_time_{pin}
            arrival_col = f'arrival_{pin}'
            arrival_time_col = f'arrival_time_{pin}'
            if arrival_col in row.index and pd.notna(row[arrival_col]):
                arrival_val = row[arrival_col]
            elif arrival_time_col in row.index and pd.notna(row[arrival_time_col]):
                arrival_val = row[arrival_time_col]
            else:
                arrival_val = 0.0

            # 逻辑值：切换引脚用推断的切换前状态，其他引脚设为 0.5（未知）
            logic_val = switching_before if pin == switching else 0.5
            feat = [
                logic_val,
                1.0 if pin == switching else 0.0,
                slew_val,
                arrival_val,
                load_val,
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
        pin_loads_dict = self.static_df.loc[cid, 'pin_loads_dict']
        dyn_feats = self._get_dynamic_features(row, pin_loads_dict)

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
        data.switching_pin = row['switching_pin']
        return data