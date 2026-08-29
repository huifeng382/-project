import pandas as pd
import numpy as np
import torch
import json
import os
from torch_geometric.data import Data, Dataset
from src.graph_builder import build_static_graph, GATE_TYPES
from src.utils import load_scaler
import config

# 模块级共享图 LRU（跨本进程所有 DelayDataset 实例；上限 GRAPH_CACHE_MAX，超限逐出最旧、磁盘回源）。
# 背景：_prepare_static_graphs 曾把全部分区图驻留内存（43.7k 图/run），4 run 并发 OOM（2026-08-28）。
_GRAPH_LRU = {}

class DelayDataset(Dataset):
    def __init__(self, static_parquets, dynamic_parquets, circuit_ids=None, scaler=None, cache_dir="cache",
                 dynamic_df=None, prefiltered=False):
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

            # vector 列存在则保留原样（V1 固定 5 位 / V2 任意 N 位，禁止 zfill 改写——会移位破坏位到引脚映射）
            if 'vector' in df.columns:
                df['vector'] = df['vector'].astype(str)

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

        # 动态数据：传入已过滤 df 则直接使用（16.4.0 内存修复：避免全量重读 parquet，
        # transistor_wave_json 列实测占动态 df ~93% 内存，此前每 run 持有 5 份 + 3 次重读堆残留 ≈ 41GB）
        # prefiltered=True 时调用方已按 circuit_ids 过滤并完成清洗，直接引用不再复制
        if dynamic_df is not None:
            self.dynamic_df = dynamic_df
            if not prefiltered:
                if circuit_ids is not None:
                    self.dynamic_df = self.dynamic_df[self.dynamic_df['circuit_id'].isin(circuit_ids)].reset_index(drop=True)
                if 'DELAY' in self.dynamic_df.columns:
                    self.dynamic_df = self.dynamic_df[self.dynamic_df['DELAY'] > 1e-12].reset_index(drop=True)
        else:
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

        # 固定 wave 行掩蔽（WAVE_COVERAGE<1 时模拟部分仿真；跨 epoch 不变）
        self._wave_mask = None
        if config.USE_TRANSISTOR_WAVE and config.WAVE_COVERAGE < 1.0:
            self._wave_mask = self._build_wave_mask()

        # 确保 vector 列格式正确（保持原样，不 zfill）
        if 'vector' in self.dynamic_df.columns:
            self.dynamic_df['vector'] = self.dynamic_df['vector'].astype(str)

        # 组ID：同 (expr,corner,switching_pin,direction,vector) = 同功能同激励下的不同变体（成对排序用）
        def _col(name):
            return self.dynamic_df[name].astype(str) if name in self.dynamic_df.columns \
                else pd.Series([''] * len(self.dynamic_df))
        _gk = (_col('expr') + '|' + _col('corner') + '|' + _col('switching_pin')
               + '|' + _col('direction') + '|' + _col('vector'))
        self.group_ids = pd.factorize(_gk)[0].tolist()        # 从静态数据中动态推断输入引脚（替代硬编码）
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

        # ---- 每电路引脚（V2 任意 I/O：不同电路引脚集合不同，按电路取，不用全局并集）----
        # self._circuit_pins[cid] = (input_pins 列表, output_pins 列表)
        self._circuit_pins = {}
        for cid, srow in self.static_df.iterrows():
            ip, op = None, None
            try:
                ip = json.loads(srow['input_pins_json']) if isinstance(srow['input_pins_json'], str) else srow['input_pins_json']
            except Exception:
                ip = None
            try:
                op = json.loads(srow['output_pins_json']) if isinstance(srow['output_pins_json'], str) else srow['output_pins_json']
            except Exception:
                op = None
            if not isinstance(ip, list) or not ip or not isinstance(op, list) or not op:
                # 从网表头推导（V1 兼容：.SUBCKT DUT 后的非电源引脚；输出=sink net）
                nl = str(srow.get('gate_level_netlist', ''))
                header = []
                for line in nl.split('\n'):
                    s = line.strip()
                    if s.upper().startswith('.SUBCKT DUT'):
                        parts = s.split()
                        if len(parts) >= 3:
                            header = [p for p in parts[2:] if p.lower() not in ('vdd', 'gnd', 'vss')]
                        break
                if not isinstance(ip, list) or not ip:
                    ip = header
                if not isinstance(op, list) or not op:
                    # sink net = 某门输出但从未被任何门当输入
                    outs = set(); consumed = set()
                    for line in nl.split('\n'):
                        s = line.strip()
                        if not s.startswith('X_'):
                            continue
                        t = s.split()
                        if len(t) < 3:
                            continue
                        outs.add(t[-2]); consumed.update(t[1:-2])
                    op = [o for o in outs if o not in consumed] or header
            self._circuit_pins[str(cid)] = (list(ip), list(op))

        self.scaler = scaler
        self.cache_dir = cache_dir
        self.graph_cache = {}
        self._graph_cache_dir = os.path.join(cache_dir, 'graphs')
        self._gate_cache_dir = os.path.join(cache_dir, 'gate')
        os.makedirs(self._graph_cache_dir, exist_ok=True)
        os.makedirs(self._gate_cache_dir, exist_ok=True)

        # 电路签名：从静态数据提取，训练/推理一致
        self._circuit_sig = {}
        for cid in self.static_df.index:
            row = self.static_df.loc[cid]
            nl = row['gate_level_netlist']
            n_gates = len([l for l in str(nl).split('\n') if l.strip().startswith('X_')])
            sig = [
                float(n_gates),
                float(row.get('transistor_count', 0)),
                float(len(json.loads(row['input_pins_json']))),
            ]
            self._circuit_sig[cid] = np.array(sig, dtype=np.float32)

        self._prepare_static_graphs()

    def _build_wave_mask(self):
        """固定 wave 行掩蔽：每电路 (switching_pin, direction) 组合按 WAVE_COVERAGE 比例保留 wave。
        确定性（seed 由 WAVE_COVERAGE_SEED + circuit_id 派生），跨 epoch 不变；
        对齐 Rust 推理「部分仿真只跑部分 (pin,dir) 向量」的场景。"""
        import hashlib, random
        mask = {}
        df = self.dynamic_df
        if 'switching_pin' not in df.columns or 'direction' not in df.columns:
            return None
        for cid, grp in df.groupby('circuit_id'):
            combos = grp[['switching_pin', 'direction']].drop_duplicates().values.tolist()
            if not combos:
                continue
            h = hashlib.md5(f"{config.WAVE_COVERAGE_SEED}|{cid}".encode()).hexdigest()[:8]
            rng = random.Random(int(h, 16))
            rng.shuffle(combos)
            keep = max(1, int(round(len(combos) * config.WAVE_COVERAGE)))
            for i, (p, d) in enumerate(combos):
                mask[(str(cid), str(p), str(d))] = i < keep
        return mask

    def _prepare_static_graphs(self):
        # 只确保磁盘图缓存齐全（缺失的构建到磁盘），不驻留内存（LRU 在 _get_static 按需加载）
        for cid in self.dynamic_df['circuit_id'].unique():
            cache_path = os.path.join(self._graph_cache_dir, f"{cid}_graph.pt")
            if not os.path.exists(cache_path):
                netlist = self.static_df.loc[cid, 'gate_level_netlist']
                ip, op = self._circuit_pins.get(str(cid), (None, None))
                node_names, node_static, edge_index = build_static_graph(cid, netlist, ip or None, op or None)
                torch.save((node_names, node_static, edge_index), cache_path)

    def _get_static(self, cid):
        # 模块级 LRU：命中刷新；未命中磁盘加载；超限逐出最旧
        c = _GRAPH_LRU.get(cid)
        if c is not None:
            _GRAPH_LRU.pop(cid)
            _GRAPH_LRU[cid] = c
            return c
        cache_path = os.path.join(self._graph_cache_dir, f"{cid}_graph.pt")
        c = torch.load(cache_path, weights_only=False)
        _GRAPH_LRU[cid] = c
        if len(_GRAPH_LRU) > config.GRAPH_CACHE_MAX:
            oldest = next(iter(_GRAPH_LRU))
            del _GRAPH_LRU[oldest]
        return c

    def _get_dynamic_features(self, row, pin_loads_dict, pins=None):
        pins = pins if pins is not None else self.pins
        switching = row['switching_pin']
        direction = row['direction']
        # 从 direction 推断 switching_pin 的切换前状态
        # rise: 0 -> 1，切换前为 0; fall: 1 -> 0，切换前为 1
        switching_before = 0.0 if direction == 'rise' else 1.0

        # V2 per-pin JSON 列（任意 I/O 用；V1 无此列时回退固定列）
        pin_slew, pin_load = {}, {}
        try:
            _v = row.get('pin_slew_json')
            _v = json.loads(_v) if isinstance(_v, str) else _v
            if isinstance(_v, dict):
                pin_slew = {str(k): v for k, v in _v.items()}
        except Exception:
            pass
        try:
            _v = row.get('pin_load_json')
            _v = json.loads(_v) if isinstance(_v, str) else _v
            if isinstance(_v, dict):
                pin_load = {str(k): v for k, v in _v.items()}
        except Exception:
            pass

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

        # vector 编码：N 位字符串，第 i 位对应第 i 个输入引脚（INORDER 序，不 zfill）
        vector_str = str(row.get('vector', ''))

        dyn_feats = {}
        for pin in pins:
            # 负载：V2 优先 pin_load_json；否则固定列 load_{pin}；否则静态字典
            if pin in pin_load:
                load_val = pin_load[pin]
            else:
                pl = pin.lower()
                if pl.startswith('out'):
                    load_val = global_out_load
                else:
                    load_col = f'load_{pin}'
                    if load_col in row.index and pd.notna(row[load_col]):
                        load_val = row[load_col]
                    else:
                        load_val = pin_loads_dict.get(pin, 0.0)

            # slew：V2 优先 pin_slew_json；否则固定列；否则仅切换引脚用全局 slew
            if pin in pin_slew:
                slew_val = pin_slew[pin]
            else:
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
                0.0,  # gate_state: 输入引脚固定为0，门节点在 __getitem__ 中设置
            ]
            if self.scaler is not None:
                # 缩放连续值特征: slew, load, out_load, arrival
                continuous = np.array([feat[2], feat[3], feat[4], feat[5]]).reshape(1, -1)
                scaled_cont = self.scaler.transform(continuous)[0]
                feat[2], feat[3], feat[4], feat[5] = (
                    scaled_cont[0], scaled_cont[1], scaled_cont[2], scaled_cont[3])
            dyn_feats[pin] = feat

        # corner 条件作为图级特征，不混入节点特征
        corner_cond = np.array([corner_slew_cond, corner_load_cond], dtype=np.float32)
        return dyn_feats, corner_cond

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        else:
            idx = int(idx)
        row = self.dynamic_df.iloc[idx]
        cid = row['circuit_id']
        node_names, node_static, edge_index = self._get_static(cid)
        pin_loads_dict = self.static_df.loc[cid, 'pin_loads_dict']
        circuit_pins, circuit_outputs = self._circuit_pins.get(str(cid), (self.pins, []))
        dyn_feats, corner_cond = self._get_dynamic_features(row, pin_loads_dict, circuit_pins)

        num_nodes = len(node_names)
        num_dyn_feats = 7
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
            vector_str = str(row.get('vector', ''))
            sw = row['switching_pin']
            gate_cache_path = os.path.join(self._gate_cache_dir,
                                            f"{cid}_{vector_str}_{sw}_gate.pt")
            compute_gate = True
            if os.path.exists(gate_cache_path):
                try:
                    with open(gate_cache_path, 'r') as f:
                        gate_states = json.load(f)
                    # json.load 可能返回空（文件损坏），加检查
                    if gate_states and isinstance(gate_states, dict) and len(gate_states) > 0:
                        compute_gate = False
                except (json.JSONDecodeError, IOError):
                    pass
            if compute_gate:
                from src.logic_sim import compute_gate_states
                from src.graph_builder import GATE_TYPES
                node_types = {}
                for j, n in enumerate(node_names):
                    type_idx = int(node_static[j, 0].item())
                    node_types[n] = GATE_TYPES[type_idx] if type_idx < len(GATE_TYPES) else 'UNKNOWN'
                gate_states = compute_gate_states(node_names, node_types, edge_index,
                                                   vector_str, circuit_pins, sw)
                try:
                    with open(gate_cache_path, 'w') as f:
                        json.dump(gate_states, f)
                except IOError:
                    pass

        gate_states_lc = {str(k).lower(): v for k, v in gate_states.items()} if isinstance(gate_states, dict) else {}
        for i, n in enumerate(node_names):
            key = str(n).lower()
            if key in gate_states_lc:
                x[i, -1] = float(gate_states_lc[key])
            elif n in circuit_outputs:
                x[i, -1] = 1.0

        # ---- 新物理特征（delivery1 提供，config 开关控制，均作为额外节点特征）----
        extra_feats = []
        # Parasitic caps: 每门总寄生电容(fF) -> 1 特征
        if config.USE_PARASITIC_CAPS:
            pc_feat = torch.zeros(num_nodes, 1)
            try:
                pc_json = self.static_df.loc[cid, 'parasitic_caps_json']
                pc = json.loads(pc_json) if isinstance(pc_json, str) else pc_json
                pc = {str(k).lower(): v for k, v in pc.items()} if isinstance(pc, dict) else {}
                for i, n in enumerate(node_names):
                    v = pc.get(str(n).lower())
                    if isinstance(v, dict):
                        pc_feat[i, 0] = sum(float(x) for x in v.values() if x is not None)
            except Exception:
                pass
            extra_feats.append(pc_feat)
        # Transistor wave: WAVE_FIELDS 按 gate 聚合 -> 节点特征（子字段选择 + 行覆盖率掩蔽）
        if config.USE_TRANSISTOR_WAVE:
            tw_dim = (3 if config.WAVE_AGG_RICH else 1) * len(config.WAVE_FIELDS)
            tw_feat = torch.zeros(num_nodes, tw_dim)
            try:
                wave_ok = True
                if self._wave_mask is not None:
                    key = (str(row.get('circuit_id')), str(row.get('switching_pin')), str(row.get('direction')))
                    wave_ok = bool(self._wave_mask.get(key, False))
                if wave_ok:
                    tw = row.get('transistor_wave_json')
                    tw = json.loads(tw) if isinstance(tw, str) else tw
                    gate_agg = {}
                    if isinstance(tw, dict):
                        for _, td in tw.items():
                            if not isinstance(td, dict): continue
                            g = str(td.get('gate', '')).lower()
                            if not g: continue
                            if g not in gate_agg:
                                gate_agg[g] = {f: [] for f in config.WAVE_FIELDS}
                            for f in config.WAVE_FIELDS:
                                v = td.get(f)
                                if v is not None: gate_agg[g][f].append(float(v))
                    import numpy as np
                    for i, n in enumerate(node_names):
                        gkey = str(n).lower()
                        if gkey in gate_agg:
                            for fi, f in enumerate(config.WAVE_FIELDS):
                                vals = gate_agg[gkey][f]
                                if not vals: continue
                                arr = np.array(vals)
                                tw_feat[i, fi] = arr.mean()
                                if config.WAVE_AGG_RICH:
                                    tw_feat[i, len(config.WAVE_FIELDS)+fi] = arr.max()
                                    tw_feat[i, 2*len(config.WAVE_FIELDS)+fi] = arr.std()
            except Exception:
                pass
            extra_feats.append(tw_feat)
        # Supply noise: vdd_droop_mV / gnd_bounce_mV 广播到所有节点 -> 2 特征
        if config.USE_SUPPLY_NOISE:
            sn_feat = torch.zeros(num_nodes, 2)
            try:
                sn = row.get('supply_noise_json')
                sn = json.loads(sn) if isinstance(sn, str) else sn
                if isinstance(sn, dict):
                    sn_feat[:, 0] = float(sn.get('vdd_droop_mV', 0))
                    sn_feat[:, 1] = float(sn.get('gnd_bounce_mV', 0))
            except Exception:
                pass
            extra_feats.append(sn_feat)
        if extra_feats:
            x = torch.cat([x] + extra_feats, dim=1)

        y = torch.tensor([row['DELAY']], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.switching_pin = row['switching_pin']
        data.corner_cond = torch.tensor(corner_cond, dtype=torch.float).unsqueeze(0)
        data.circuit_sig = torch.tensor(self._circuit_sig.get(cid, [0,0,0]),
                                         dtype=torch.float).unsqueeze(0)
        # 结构先验（transistor_count + 门类型计数）
        if config.USE_STRUCT_PRIOR:
            try:
                ct_json = self.static_df.loc[cid, 'cell_types_json']
                ct = json.loads(ct_json) if isinstance(ct_json, str) else ct_json
                ct = ct if isinstance(ct, list) else []
                data.struct_prior = torch.tensor([
                    float(self.static_df.loc[cid, 'transistor_count']),
                    float(sum(1 for g in ct if 'SC_AND' in str(g) and 'SC_AND_' not in str(g))),
                    float(sum(1 for g in ct if 'SC_INV_WIRE' in str(g))),
                ], dtype=torch.float).unsqueeze(0)
            except Exception:
                data.struct_prior = torch.zeros(1, 3)

        # 逐门监督标签（per_gate_timing_json，按节点名对齐，与 gate_states 同一套 node_names 顺序）
        # 缺失字段填 -1（哨兵值，loss 侧用 >0 过滤）。单位为 ps。
        pg_delay = torch.full((num_nodes,), -1.0, dtype=torch.float)
        pg_out_slew = torch.full((num_nodes,), -1.0, dtype=torch.float)
        pg_in_slew = torch.full((num_nodes,), -1.0, dtype=torch.float)
        pgt = row.get('per_gate_timing_json')
        if isinstance(pgt, str):
            try:
                pgt = json.loads(pgt)
            except Exception:
                pgt = None
        if isinstance(pgt, dict):
            pgt_lc = {str(k).lower(): v for k, v in pgt.items()}
            for i, n in enumerate(node_names):
                v = pgt_lc.get(str(n).lower())
                if isinstance(v, dict):
                    d, o, s = v.get('delay_ps'), v.get('out_slew_ps'), v.get('in_slew_ps')
                    if d is not None: pg_delay[i] = float(d)
                    if o is not None: pg_out_slew[i] = float(o)
                    if s is not None: pg_in_slew[i] = float(s)
        data.per_gate_delay = pg_delay
        data.per_gate_out_slew = pg_out_slew
        data.per_gate_in_slew = pg_in_slew
        # 组ID（成对排序损失用）：同组=同功能同激励的不同变体
        data.grp = torch.tensor([self.group_ids[idx]], dtype=torch.long)
        # 行号（蒸馏用）：teacher 预测 npz 按 dataset 行 idx 对齐（Subset 离群清洗后仍指向原 idx）
        data.row_idx = torch.tensor([idx], dtype=torch.long)
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
        pins = self._circuit_pins.get(str(cid), (self.pins, []))[0]
        
        # ----- 1. 图级静态统计 -----
        fanout = node_static_np[:, 1]  # 绝对索引（静态特征位置固定）
        depth = node_static_np[:, 2]
        drive = node_static_np[:, 3]
        
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

        out_nodes = self._circuit_pins.get(str(cid), (None, ['out']))[1] or ['out']
        dist_to_out = {n: float('inf') for n in node_names}
        q = deque()
        for on in out_nodes:
            if on in node_names:
                dist_to_out[on] = 0
                q.append(on)
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
                        fanout_vals.append(node_static_np[idx_n, 1])
                        drive_vals.append(node_static_np[idx_n, 3])
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