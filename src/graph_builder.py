import torch
import numpy as np
from collections import deque
import re

# 定义门类型映射（根据实际网表中出现的门类型扩充）
GATE_TYPES = [
    # 原有 batch01 的门类型
    'SC_AND', 'SC_AND_v1',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE',
    'SC_INV_WIRE', 'INPUT_PIN', 'OUTPUT_PIN',
    'SC_JOIN_BRIDGE__BRIDGE', 
    # batch02 新增的门类型
    'AND2x2_ASAP7_75t_R',
    'AND3x2_ASAP7_75t_R',
    'BUFx1_ASAP7_75t_R',
    'INVx1_ASAP7_75t_R',
    'NAND3x1_ASAP7_75t_R',
    'NAND5x1_ASAP7_75t_R',
    'NOR2x1_ASAP7_75t_R',
    'NOR3x1_ASAP7_75t_R',
    'NOR4x1_ASAP7_75t_R',
    'NOR5x1_ASAP7_75t_R',
    'OR2x2_ASAP7_75t_R',
    'OR3x2_ASAP7_75t_R',
    'TIEHIx1_ASAP7_75t_R',
    'TIELOx1_ASAP7_75t_R',
    # batch06 新增的 ASAP7 标准单元
    'AND8x2_ASAP7_75t_R',
    'NAND4x1_ASAP7_75t_R',
    'OR8x2_ASAP7_75t_R',
    'UNKNOWN_GATE',
]
GATE_TO_IDX = {gt: i for i, gt in enumerate(GATE_TYPES)}


def _normalize_gate_type(gt):
    """将长hash名归一化为基础门类型，让每种门有足够样本学习embedding"""
    base = str(gt)
    # 保留特殊类型
    if base in ('INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE'):
        return base
    # ASAP7标准单元：提取基础门类型 AND/OR/INV/NAND/NOR/BUF
    if 'ASAP7' in base:
        for kw in ['NAND', 'NOR', 'XOR', 'AND', 'OR', 'INV', 'BUF', 'TIEHI', 'TIELO']:
            if kw in base:
                return kw
        return 'OTHER'
    # SC_系列：提取最后一个有意义的门类型关键词
    if base.startswith('SC_'):
        # 基础门类型（短名，不转换）
        simple = {'SC_AND', 'SC_OR', 'SC_NAND', 'SC_NOR', 'SC_INV', 'SC_XOR',
                  'SC_BUF', 'SC_JOIN', 'SC_BRIDGE', 'SC_INV_WIRE',
                  'SC_AND_v1', 'SC_OR_v1', 'SC_BRIDGE_v1', 'SC_JOIN_v1'}
        if base in simple:
            return base
        # 长hash名：提取最后的门功能关键词
        for kw in ['NAND', 'NOR', 'XOR', 'AND', 'OR', 'INV', 'BUF']:
            if kw in base:
                return f'SC_{kw}'
        # JOIN/BRIDGE/WIRE类：归为JOIN
        if 'JOIN' in base or 'BRIDGE' in base or 'WIRE' in base:
            return 'SC_JOIN'
    return base


def rebuild_gate_types(cell_types):
    """
    从实际数据中动态构建门类型映射。
    长hash门名归一化为基础类型，让每种门有足够样本学习embedding。
    """
    global GATE_TYPES, GATE_TO_IDX
    reserved = ['INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE']
    GATE_TYPES = sorted(set(_normalize_gate_type(ct) for ct in cell_types)) + reserved
    GATE_TO_IDX = {gt: i for i, gt in enumerate(GATE_TYPES)}

def parse_netlist(netlist_str):
    lines = netlist_str.strip().split('\n')
    gates = {}
    wire_to_driver = {}
    input_pins = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith('.SUBCKT DUT'):
            parts = stripped.split()
            if len(parts) >= 3:
                input_pins = [
                    p for p in parts[2:]
                    if p.lower() not in ('vdd', 'gnd', 'vss', 'out')
                ]
            continue
        if not stripped.startswith('X_'):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue
        inst = tokens[0]
        # SPICE 实例行格式: X_<inst> <nets...> <subckt_name>
        # 最后一个 token 是 subckt 名称（门类型）
        gtype = tokens[-1]
        # 中间的 nets：去掉实例名和门类型
        io_tokens = tokens[1:-1]
        if len(io_tokens) < 1:
            continue
        # 按 SPICE 惯例，最后一个 net 是输出，前面的是输入
        output = io_tokens[-1]
        inputs = io_tokens[:-1]
        gates[inst] = {'type': gtype, 'inputs': inputs, 'output': output}
        wire_to_driver[output] = inst

    nodes = {}
    for inst, info in gates.items():
        nodes[inst] = {'type': info['type'], 'is_input': False, 'is_output': False}

    for pin in input_pins:
        nodes[pin] = {'type': 'INPUT_PIN', 'is_input': True, 'is_output': False}
    nodes['out'] = {'type': 'OUTPUT_PIN', 'is_input': False, 'is_output': True}

    edges = []
    for inst, info in gates.items():
        for inp in info['inputs']:
            if inp in input_pins:
                edges.append((inp, inst))
    for inst, info in gates.items():
        for inp in info['inputs']:
            if inp in wire_to_driver and wire_to_driver[inp] != inst:
                driver_inst = wire_to_driver[inp]
                edges.append((driver_inst, inst))
    for inst, info in gates.items():
        if info['output'] == 'out':
            edges.append((inst, 'out'))

    edges = list(set(edges))
    return nodes, edges

def build_static_graph(circuit_id, netlist_str):
    nodes, edges = parse_netlist(netlist_str)
    node_names = list(nodes.keys())
    
    # 如果没有边，添加自环边
    if len(edges) == 0:
        edges = [(n, n) for n in node_names]
        print(f"WARNING: circuit {circuit_id} had no edges, added self-loops")
    
    # 构建 edge_index
    edge_index = []
    for u, v in edges:
        if u in node_names and v in node_names:
            edge_index.append([node_names.index(u), node_names.index(v)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 节点门类型索引（不再用 one-hot，改为整数索引，由模型 Embedding 层处理）
    node_type_idx = []
    for n in node_names:
        gt = nodes[n]['type']
        idx = GATE_TO_IDX.get(_normalize_gate_type(gt), GATE_TO_IDX['UNKNOWN_GATE'])
        node_type_idx.append([float(idx)])
    node_type_idx = torch.tensor(node_type_idx, dtype=torch.float)
    
    # 1. 扇出数（出度）
    out_degree = {n: 0 for n in node_names}
    for u, v in edges:
        out_degree[u] += 1
    
    # 2. 逻辑深度（最长路径，拓扑排序）
    adj = {n: [] for n in node_names}
    indeg = {n: 0 for n in node_names}
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque([n for n in node_names if indeg[n] == 0])
    depth = {n: 0 for n in node_names}
    while q:
        u = q.popleft()
        for v in adj[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    # 3. 驱动强度
    def get_drive_strength(gate_type):
        match = re.search(r'x(\d+)', gate_type, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'AND(\d+)x', gate_type, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 1.0

    # 4. 逻辑努力参数（寄生延迟 p, 逻辑努力 g, 输入电容 C_in）
    def get_logic_params(gate_type, num_inputs):
        gt = gate_type.upper()
        n = max(num_inputs, 1)
        # 寄生延迟 p
        if 'INV' in gt or 'NOT' in gt:    p = 1.0
        elif 'NAND' in gt:                p = n
        elif 'NOR' in gt:                 p = n
        elif 'AND' in gt:                 p = n + 1   # AND = NAND + INV
        elif 'OR' in gt:                  p = n + 1   # OR = NOR + INV
        elif 'XOR' in gt:                 p = 2 * n
        elif 'BUF' in gt:                 p = 1.0
        elif 'JOIN' in gt or 'BRIDGE' in gt or 'WIRE' in gt: p = 0.5
        else:                             p = 1.0
        # 逻辑努力 g
        if 'INV' in gt or 'NOT' in gt:    g = 1.0
        elif 'NAND' in gt:                g = (n + 2) / 3
        elif 'NOR' in gt:                 g = (2 * n + 1) / 3
        elif 'AND' in gt:                 g = (n + 2) / 3   # NAND part
        elif 'OR' in gt:                  g = (2 * n + 1) / 3  # NOR part
        elif 'XOR' in gt:                 g = 4.0
        elif 'BUF' in gt:                 g = 1.0
        elif 'JOIN' in gt or 'BRIDGE' in gt or 'WIRE' in gt: g = 0.5
        else:                             g = 1.0
        # 输入电容 C_in (从门类型名提取驱动等级)
        match = re.search(r'x(\d+)', gate_type, re.IGNORECASE)
        cin = float(match.group(1)) if match else 1.0
        return p, g, cin

    drive_strength = []
    parasitic = []
    logic_effort = []
    input_cap = []
    fanin_count = []
    for n in node_names:
        if n.startswith('X_'):
            gt = nodes[n]['type']
            ds = get_drive_strength(gt)
            num_in = len([e for e in edges if e[1] == n])
            p, g, cin = get_logic_params(gt, num_in)
        else:
            ds = 0.0
            num_in = 0
            p, g, cin = 0.0, 0.0, 0.0
        drive_strength.append(ds)
        parasitic.append(p)
        logic_effort.append(g)
        input_cap.append(cin)
        fanin_count.append(num_in)

    # 电努力 h = 输出负载 / 输入电容（仅门节点，引脚为0）
    out_load = [0.0] * len(node_names)
    for u, v in edges:
        ui = node_names.index(u)
        vi = node_names.index(v)
        out_load[ui] += input_cap[vi]
    electrical_effort = []
    for i, n in enumerate(node_names):
        if n.startswith('X_'):
            h = out_load[i] / max(input_cap[i], 0.01)
        else:
            h = 0.0
        electrical_effort.append(h)

    # 将特征转换为张量（使用 log1p 平滑）
    fanout_feat  = torch.tensor([[np.log1p(out_degree[n])] for n in node_names], dtype=torch.float)
    depth_feat   = torch.tensor([[np.log1p(depth[n])] for n in node_names], dtype=torch.float)
    drive_feat   = torch.tensor([[ds] for ds in drive_strength], dtype=torch.float)
    p_feat       = torch.tensor([[p] for p in parasitic], dtype=torch.float)
    g_feat       = torch.tensor([[g] for g in logic_effort], dtype=torch.float)
    h_feat       = torch.tensor([[np.log1p(h)] for h in electrical_effort], dtype=torch.float)

    # 合并静态特征：门类型索引 + 扇出 + 深度 + 驱动 + 寄生延迟 + 逻辑努力 + 电努力
    node_static = torch.cat([node_type_idx, fanout_feat, depth_feat, drive_feat,
                              p_feat, g_feat, h_feat], dim=1)

    return node_names, node_static, edge_index