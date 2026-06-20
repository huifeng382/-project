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
            
            # 确保 input_pins_json 列存在（若没有，则补默认5引脚）
            if 'input_pins_json' not in df.columns:
                df['input_pins_json'] = ['["a","b","c","d","e"]'] * len(df)
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

            # 确保 direction 列存在（若没有，默认 'rise'）
            if 'direction' not in df.columns:
                df['direction'] = 'rise'

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
            self.dynamic_df = self.dynamic_df[self.dynamic_df['circuit_id'].isin(circuit_ids)]

        # 删除 DELAY 为 NaN 的行
        self.dynamic_df = self.dynamic_df.dropna(subset=['DELAY']).reset_index(drop=True)

        # 动态读取引脚列表（从 static_df 中读取 input_pins_json）
        if 'input_pins_json' in self.static_df.columns:
            first_row = self.static_df.iloc[0]
            try:
                pins = json.loads(first_row['input_pins_json'])
                if isinstance(pins, list) and len(pins) > 0:
                    self.pins = pins
                else:
                    self.pins = ['a','b','c','d','e']
                    print("Warning: input_pins_json is not a valid list, using default pins.")
            except:
                self.pins = ['a','b','c','d','e']
                print("Warning: Failed to parse input_pins_json, using default pins.")
        else:
            self.pins = ['a','b','c','d','e']
            print("Warning: No 'input_pins_json' column, using default pins.")

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
        # 获取 switching_pin 和 direction
        switching = row['switching_pin']
        direction = row.get('direction', 'rise')

        # 确定切换引脚的逻辑值
        if direction == 'rise':   # 0->1 跳变，跳变前为 0
            switch_logic = 0.0
        elif direction == 'fall': # 1->0 跳变，跳变前为 1
            switch_logic = 1.0
        else:
            switch_logic = 0.5   # 未知方向

        # 构建逻辑值字典，非切换引脚一律设为 0.5
        logic = {pin: 0.5 for pin in pins}
        logic[switching] = switch_logic

        dyn_feats = {}
        for pin in pins:
            # 负载：优先从动态数据读取 load_{pin}，否则从静态字典
            load_col = f'load_{pin}'
            if load_col in row.index and pd.notna(row[load_col]):
                load_val = row[load_col]
            else:
                load_val = pin_loads_dict.get(pin, 0.0)

            # 获取 slew 和 arrival
            slew_col = f'slew_{pin}'
            arrival_col = f'arrival_{pin}'
            if slew_col in row.index and pd.notna(row[slew_col]):
                slew_val = row[slew_col]
            else:
                slew_val = row.get('slew_s', 0.0)
            if arrival_col in row.index and pd.notna(row[arrival_col]):
                arrival_val = row[arrival_col]
            else:
                arrival_val = row.get('arrival_time_s', 0.0)

            feat = [
                float(logic[pin]),      # 使用新的逻辑值
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