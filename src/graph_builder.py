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


def rebuild_gate_types(cell_types):
    """
    从实际数据中动态构建门类型映射。
    cell_types: 所有出现的 cell 类型字符串的可迭代集合。
    调用后 GATE_TYPES 和 GATE_TO_IDX 会被重建，包含所有 found types +
    INPUT_PIN / OUTPUT_PIN / UNKNOWN_GATE。
    """
    global GATE_TYPES, GATE_TO_IDX
    reserved = ['INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE']
    GATE_TYPES = sorted(set(cell_types)) + reserved
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
        idx = GATE_TO_IDX.get(gt, GATE_TO_IDX['UNKNOWN_GATE'])
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
    
    drive_strength = []
    for n in node_names:
        if n.startswith('X_'):
            gt = nodes[n]['type']
            ds = get_drive_strength(gt)
        else:
            ds = 0.0
        drive_strength.append(ds)
    
    # 4. 扇入数（入度）
    in_degree = {n: 0 for n in node_names}
    for u, v in edges:
        in_degree[v] += 1

    # 5. 到输出节点的最短距离（BFS 反向搜索）
    reverse_adj = {n: [] for n in node_names}
    for u, v in edges:
        reverse_adj[v].append(u)
    dist_to_out = {n: float('inf') for n in node_names}
    if 'out' in node_names:
        dist_to_out['out'] = 0
        q = deque(['out'])
        while q:
            u = q.popleft()
            for prev in reverse_adj[u]:
                if dist_to_out[prev] > dist_to_out[u] + 1:
                    dist_to_out[prev] = dist_to_out[u] + 1
                    q.append(prev)
    for n in node_names:
        if dist_to_out[n] == float('inf'):
            dist_to_out[n] = len(node_names)  # 不可达节点给最大值

    # 6. 是否在关键路径上（可达输出且可从输入到达）
    # 从所有输入引脚正向 BFS，标记可达节点
    reachable_from_input = {n: False for n in node_names}
    fwd_q = deque([n for n in node_names if nodes[n].get('is_input')])
    for n in fwd_q:
        reachable_from_input[n] = True
    while fwd_q:
        u = fwd_q.popleft()
        for v in adj.get(u, []):
            if not reachable_from_input.get(v, False):
                reachable_from_input[v] = True
                fwd_q.append(v)
    # 从输出反向 BFS，标记可达输出的节点
    reachable_to_output = {n: (dist_to_out[n] < len(node_names)) for n in node_names}
    on_path = {n: (reachable_from_input.get(n, False) and reachable_to_output.get(n, False))
               for n in node_names}

    # 将特征转换为张量（使用 log1p 平滑）
    fanout_feat  = torch.tensor([[np.log1p(out_degree[n])] for n in node_names], dtype=torch.float)
    depth_feat   = torch.tensor([[np.log1p(depth[n])] for n in node_names], dtype=torch.float)
    drive_feat   = torch.tensor([[ds] for ds in drive_strength], dtype=torch.float)
    fanin_feat   = torch.tensor([[np.log1p(in_degree[n])] for n in node_names], dtype=torch.float)
    dist_feat    = torch.tensor([[float(dist_to_out[n])] for n in node_names], dtype=torch.float)
    onpath_feat  = torch.tensor([[1.0 if on_path[n] else 0.0] for n in node_names], dtype=torch.float)

    # 合并静态特征：门类型索引 + 扇出 + 深度 + 驱动 + 扇入 + 输出距离 + 关键路径
    node_static = torch.cat([node_type_idx, fanout_feat, depth_feat, drive_feat,
                              fanin_feat, dist_feat, onpath_feat], dim=1)

    return node_names, node_static, edge_index