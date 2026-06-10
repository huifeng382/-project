import torch
import numpy as np
from collections import deque
import re

# 定义门类型映射（根据实际网表中出现的门类型扩充）
GATE_TYPES = [
    'SC_AND', 'SC_AND_v1', 'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE',
    'SC_INV_WIRE', 'INPUT_PIN', 'OUTPUT_PIN'
]
GATE_TO_IDX = {gt: i for i, gt in enumerate(GATE_TYPES)}

def parse_netlist(netlist_str):
    """解析网表，返回 nodes 字典和 edges 列表。"""
    lines = netlist_str.strip().split('\n')
    gates = {}
    wire_to_driver = {}

    for line in lines:
        if not line.startswith('X_'):
            continue
        tokens = line.split()
        inst = tokens[0]
        gtype = tokens[-1]
        io = tokens[1:-1]
        if 'AND' in gtype or 'INV_WIRE' in gtype:
            output = io[-1]
            inputs = io[:-1]
        elif 'JOIN_BRIDGE' in gtype:
            output = io[-1]
            inputs = io[:-1]
        else:
            continue
        gates[inst] = {'type': gtype, 'inputs': inputs, 'output': output}
        wire_to_driver[output] = inst

    nodes = {}
    for inst, info in gates.items():
        nodes[inst] = {'type': info['type'], 'is_input': False, 'is_output': False}

    input_pins = ['a', 'b', 'c', 'd', 'e']
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
    
    # 节点 one-hot 编码
    node_type_enc = []
    for n in node_names:
        gt = nodes[n]['type']
        onehot = [0.0] * len(GATE_TYPES)
        onehot[GATE_TO_IDX[gt]] = 1.0
        node_type_enc.append(onehot)
    node_type_enc = torch.tensor(node_type_enc, dtype=torch.float)
    
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
    
    # 将特征转换为张量（使用 log1p 平滑）
    fanout_feat = torch.tensor([[np.log1p(out_degree[n])] for n in node_names], dtype=torch.float)
    depth_feat  = torch.tensor([[np.log1p(depth[n])] for n in node_names], dtype=torch.float)
    drive_feat  = torch.tensor([[ds] for ds in drive_strength], dtype=torch.float)
    
    # 合并静态特征：one-hot + 扇出 + 深度 + 驱动强度
    node_static = torch.cat([node_type_enc, fanout_feat, depth_feat, drive_feat], dim=1)
    
    return node_names, node_static, edge_index