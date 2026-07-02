import pandas as pd
import numpy as np
import torch
import json
import os
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

            # 网表列：优先使用标准化网表（gate_level_netlist_std），
            # 它使用ASAP7标准单元名称，门类型覆盖率远高于原始网表
            if 'gate_level_netlist_std' in df.columns:
                df = df.drop(columns=['gate_level_netlist'], errors='ignore')
                df = df.rename(columns={'gate_level_netlist_std': 'gate_level_netlist'})
            elif 'gate_level_netlist' not in df.columns:
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

            # 过滤非法延迟值（<1e-12s 视为物理不可行噪声，log10 会出极端值）
            if 'DELAY' in df.columns:
                before = len(df)
                df = df[df['DELAY'] > 1e-12]
                removed = before - len(df)
                if removed > 0:
                    print(f"normalize_dynamic: removed {removed} rows with DELAY <= 1e-12")

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
        os.makedirs(self.cache_dir, exist_ok=True)
        for cid in self.dynamic_df['circuit_id'].unique():
            cache_path = os.path.join(self.cache_dir, f"{cid}_graph.pt")
            if os.path.exists(cache_path):
                # 加载缓存
                node_names, node_static, edge_index = torch.load(cache_path)
            else:
                netlist = self.static_df.loc[cid, 'gate_level_netlist']
                node_names, node_static, edge_index = build_static_graph(cid, netlist)
                torch.save((node_names, node_static, edge_index), cache_path)
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

        # 全局动态参数（来自 timing_arcs，每个向量不同）
        global_slew = row.get('slew_s', 0.0) if pd.notna(row.get('slew_s')) else 0.0
        global_out_load = row.get('output_load_f', 0.0) if pd.notna(row.get('output_load_f')) else 0.0
        global_arrival = row.get('arrival_time_s', 0.0) if pd.notna(row.get('arrival_time_s')) else 0.0

        # 解析 corner 条件（如 s05p0_l10p0 → slew=5.0ps, load=10.0fF）
        corner_str = str(row.get('corner', 's05p0_l10p0'))
        corner_slew_cond = 5.0   # 默认值
        corner_load_cond = 10.0
        try:
            s_part = corner_str.split('_')[0]  # s05p0
            l_part = corner_str.split('_')[1]  # l10p0
            corner_slew_cond = float(s_part[1:].replace('p', '.'))   # 05p0 → 5.0
            corner_load_cond = float(l_part[1:].replace('p', '.'))   # 10p0 → 10.0
        except (IndexError, ValueError):
            pass

        # vector 编码：5位字符串，每位对应一个引脚的逻辑状态
        vector_str = str(row.get('vector', '00000')).zfill(5)

        dyn_feats = {}
        for pin in pins:
            # 负载逻辑：
            # 1. 输出引脚用全局 output_load_f
            # 2. 输入引脚优先读动态列 load_{pin}，否则从静态字典
            pl = pin.lower()
            if pl.startswith('out'):
                load_val = global_out_load
            else:
                load_col = f'load_{pin}'
                if load_col in row.index and pd.notna(row[load_col]):
                    load_val = row[load_col]
                else:
                    load_val = pin_loads_dict.get(pin, 0.0)

            # 获取 slew（切换引脚有输入 slew，其他引脚优先读 per-pin 列）
            slew_col = f'slew_{pin}'
            if slew_col in row.index and pd.notna(row[slew_col]):
                slew_val = row[slew_col]
            elif pin == switching:
                slew_val = global_slew
            else:
                slew_val = 0.0

            # 获取 arrival_time（仅切换引脚有意义，非切换引脚为静态→填0）
            if pin == switching:
                arrival_col = f'arrival_time_{pin}'
                if arrival_col in row.index and pd.notna(row[arrival_col]):
                    arrival_val = row[arrival_col]
                else:
                    arrival_val = global_arrival
            else:
                arrival_val = 0.0

            # 逻辑值：切换引脚用推断的切换前状态
            # 非切换引脚：从 vector 对应位读取实际逻辑状态（不再用 0.5 占位）
            if pin == switching:
                logic_val = switching_before
            else:
                try:
                    bit_idx = pins.index(pin)
                    logic_val = float(vector_str[bit_idx]) if bit_idx < len(vector_str) else 0.5
                except (ValueError, IndexError):
                    logic_val = 0.5

            feat = [
                logic_val,
                1.0 if pin == switching else 0.0,
                slew_val,
                load_val,
                global_out_load,
                arrival_val,
                corner_slew_cond,   # corner的S条件（ps），所有引脚相同
                corner_load_cond,   # corner的L条件（fF），所有引脚相同
                0.0,  # gate_state: 输入引脚固定为0，门节点在 __getitem__ 中设置
            ]
            if self.scaler is not None:
                # 缩放连续值特征: slew, load, out_load, arrival, corner_slew, corner_load
                continuous = np.array([feat[2], feat[3], feat[4], feat[5],
                                       feat[6], feat[7]]).reshape(1, -1)
                scaled_cont = self.scaler.transform(continuous)[0]
                feat[2], feat[3], feat[4], feat[5], feat[6], feat[7] = (
                    scaled_cont[0], scaled_cont[1], scaled_cont[2], scaled_cont[3],
                    scaled_cont[4], scaled_cont[5])
            dyn_feats[pin] = feat
        return dyn_feats

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
        num_dyn_feats = 9
        node_feat_dim = node_static.shape[1] + num_dyn_feats
        x = torch.zeros((num_nodes, node_feat_dim), dtype=torch.float)
        x[:, :node_static.shape[1]] = node_static

        for i, n in enumerate(node_names):
            if n in dyn_feats:
                dyn = dyn_feats[n]
                x[i, -num_dyn_feats:] = torch.tensor(dyn, dtype=torch.float)

        # 路径特征：标记哪些门在信号路径上
        # 优先从 gate_states_json 读取，若为空则用轻量级逻辑仿真推导
        gate_states = {}
        try:
            gs = row.get('gate_states_json')
            if gs is not None and pd.notna(gs) and str(gs).strip() not in ('', '{}'):
                gate_states = json.loads(gs) if isinstance(gs, str) else gs
        except Exception:
            pass

        if not gate_states:
            # 用逻辑仿真推导信号路径
            from src.logic_sim import compute_gate_states
            from src.graph_builder import GATE_TYPES
            node_types = {}
            for j, n in enumerate(node_names):
                type_idx = int(node_static[j, 0].item())
                node_types[n] = GATE_TYPES[type_idx] if type_idx < len(GATE_TYPES) else 'UNKNOWN'
            vector_str = str(row.get('vector', '00000')).zfill(5)
            gate_states = compute_gate_states(node_names, node_types, edge_index,
                                               vector_str, self.pins, row['switching_pin'])

        for i, n in enumerate(node_names):
            if n in gate_states:
                x[i, -1] = float(gate_states[n])
            elif n == 'out':
                x[i, -1] = 1.0

        y = torch.tensor([row['DELAY']], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.switching_pin = row['switching_pin']
        return data
    def extract_features(self, idx):
        """
        提取第 idx 个样本的特征向量和标签（用于 XGBoost 等树模型）
        返回: (features: np.ndarray, label: float)
        """
        row = self.dynamic_df.iloc[idx]
        cid = row['circuit_id']
        node_names, node_static, edge_index = self._get_static(cid)
        node_static_np = node_static.numpy()
        num_nodes = node_static_np.shape[0]
        
        # ----- 1. 图级静态统计 -----
        fanout = node_static_np[:, -3]
        depth = node_static_np[:, -2]
        drive = node_static_np[:, -1]
        
        features = []
        num_edges = edge_index.size(1)
        features.extend([num_nodes, num_edges])
        features.extend([np.mean(fanout), np.max(fanout), np.std(fanout)])
        features.extend([np.mean(depth), np.max(depth), np.std(depth)])
        features.extend([np.mean(drive), np.max(drive), np.std(drive)])
        
        # ----- 2. 动态特征：切换引脚、方向 -----
        switching = row['switching_pin']
        direction = row['direction']
        for p in self.pins:
            features.append(1.0 if p == switching else 0.0)
        features.append(0.0 if direction == 'rise' else 1.0)

        # 全局动态参数（每个样本不同）
        features.append(float(row.get('slew_s', 0.0)))         # 全局 slew
        features.append(float(row.get('output_load_f', 0.0)))  # 输出负载

        # vector 编码（5-bit 输入模式，决定不同 delay 的关键特征）
        vector = str(row.get('vector', '00000')).zfill(5)
        for bit in vector:
            features.append(float(bit))
        # vector 数值归一化（0~31 → 0~1）
        try:
            features.append(int(vector) / 31.0)
        except ValueError:
            features.append(0.0)

        # ----- 3. 各引脚的 slew/load 的统计量 -----
        slew_vals = []
        load_vals = []
        # 获取该样本对应的静态负载字典
        pin_loads_dict = self.static_df.loc[cid, 'pin_loads_dict']
        for p in self.pins:
            # 读取 slew（优先 per-pin 列，否则只有切换引脚用全局 slew）
            slew = row.get(f'slew_{p}')
            if slew is not None and not pd.isna(slew):
                slew_val = float(slew)
            elif p == switching:
                slew_val = float(row.get('slew_s', 0.0))
            else:
                slew_val = 0.0
            slew_vals.append(slew_val)

            # 从静态字典读取负载，不存在则用 0.0
            load_val = pin_loads_dict.get(p, 0.0)
            load_vals.append(float(load_val) if not pd.isna(load_val) else 0.0)

        # ----- 4. 切换引脚的单独特征 -----
        if switching in self.pins:
            sw_idx = self.pins.index(switching)
            features.extend([slew_vals[sw_idx], load_vals[sw_idx]])
        else:
            features.extend([0.0, 0.0])
        
        # ----- 5. 路径级特征（添加防御性处理）-----
        from collections import deque
        reverse_adj = {n: [] for n in node_names}
        for u, v in edge_index.t().tolist():
            if u < len(node_names) and v < len(node_names):
                reverse_adj[node_names[v]].append(node_names[u])
        
        dist_to_out = {n: float('inf') for n in node_names}
        dist_to_out['out'] = 0
        q = deque(['out'])
        while q:
            u = q.popleft()
            for prev in reverse_adj.get(u, []):
                if dist_to_out[prev] > dist_to_out[u] + 1:
                    dist_to_out[prev] = dist_to_out[u] + 1
                    q.append(prev)
        
        # 收集输入引脚距离（替换 inf 为 0）
        input_pins = [p for p in self.pins if p in node_names]
        if input_pins:
            path_lengths = []
            for p in input_pins:
                d = dist_to_out.get(p, 0.0)
                if np.isinf(d):
                    d = 0.0
                path_lengths.append(d)
            features.extend([
                np.mean(path_lengths),
                np.std(path_lengths) if len(path_lengths) > 1 else 0.0,
                np.max(path_lengths),
                np.min(path_lengths),
                np.median(path_lengths)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 路径上平均扇出和驱动强度
        fanout_vals = []
        drive_vals = []
        for pin in input_pins:
            if dist_to_out.get(pin, float('inf')) < float('inf'):
                path_nodes = [n for n in node_names if dist_to_out.get(n, float('inf')) <= dist_to_out[pin]]
                for n in path_nodes:
                    if n in node_names:
                        idx_n = node_names.index(n)
                        fanout_vals.append(node_static_np[idx_n, -3])
                        drive_vals.append(node_static_np[idx_n, -1])
        if fanout_vals:
            features.extend([np.mean(fanout_vals), np.std(fanout_vals), np.max(fanout_vals)])
        else:
            features.extend([0.0, 0.0, 0.0])
        if drive_vals:
            features.extend([np.mean(drive_vals), np.std(drive_vals), np.max(drive_vals)])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # ----- 清理所有特征，确保无 NaN 或 Inf -----
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        label = row['DELAY']
        return features, label
    

    def __len__(self):
        return len(self.dynamic_df)