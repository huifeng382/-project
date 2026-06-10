# graph_builder.py
import re
import torch
from collections import deque

# 定义门类型映射（根据实际网表中出现的门类型扩充）
GATE_TYPES = [
    'SC_AND', 'SC_AND_v1', 'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE_WIRE',
    'SC_JOIN_BRIDGE_WIRE_WIRE_WIRE_BRIDGE_WIRE_WIRE_WIRE',
    'SC_INV_WIRE', 'INPUT_PIN', 'OUTPUT_PIN'
]
GATE_TO_IDX = {gt: i for i, gt in enumerate(GATE_TYPES)}

def parse_netlist(netlist_str):
    """
    解析网表，返回 nodes 字典和 edges 列表。
    nodes: {node_name: {'type': gate_type, 'is_input': bool, 'is_output': bool}}
    edges: [(from_node, to_node)]
    """
    lines = netlist_str.strip().split('\n')
    gates = {}          # inst_name -> {'type':..., 'inputs': [...], 'output': wire}
    wire_to_driver = {} # wire -> inst_name that drives it

    # 第一遍：提取所有门及其输入输出
    for line in lines:
        if not line.startswith('X_'):
            continue
        tokens = line.split()
        inst = tokens[0]
        gtype = tokens[-1]
        io = tokens[1:-1]  # 所有非 vdd/gnd 的引脚/连线
        # 根据门类型推断输出（最后一个元素通常是输出连线）
        if 'AND' in gtype or 'INV_WIRE' in gtype:
            output = io[-1]
            inputs = io[:-1]
        elif 'JOIN_BRIDGE' in gtype:
            output = io[-1]
            inputs = io[:-1]
        else:
            continue   # 未知类型跳过
        gates[inst] = {'type': gtype, 'inputs': inputs, 'output': output}
        wire_to_driver[output] = inst

    # 构建节点集合
    nodes = {}
    # 添加门节点
    for inst, info in gates.items():
        nodes[inst] = {'type': info['type'], 'is_input': False, 'is_output': False}

    # 添加输入引脚节点 (a, b, c, d, e)
    input_pins = ['a', 'b', 'c', 'd', 'e']
    for pin in input_pins:
        nodes[pin] = {'type': 'INPUT_PIN', 'is_input': True, 'is_output': False}

    # 添加输出节点
    nodes['out'] = {'type': 'OUTPUT_PIN', 'is_input': False, 'is_output': True}

    # 构建边
    edges = []
    # 输入引脚到第一级门的边
    for inst, info in gates.items():
        for inp in info['inputs']:
            if inp in input_pins:
                edges.append((inp, inst))
    # 门之间的边（根据 wire 连接）
    for inst, info in gates.items():
        for inp in info['inputs']:
            if inp in wire_to_driver and wire_to_driver[inp] != inst:
                driver_inst = wire_to_driver[inp]
                edges.append((driver_inst, inst))
    # 最后一级门到输出节点的边
    for inst, info in gates.items():
        if info['output'] == 'out':
            edges.append((inst, 'out'))
    # 去重（可选）
    edges = list(set(edges))
    return nodes, edges

def build_static_graph(circuit_id, netlist_str):
    """构建静态图：返回节点列表（按固定顺序）、节点类型编码（one-hot）、边索引"""
    nodes, edges = parse_netlist(netlist_str)
    node_names = list(nodes.keys())  # 固定顺序
    node_type_enc = []
    for n in node_names:
        gt = nodes[n]['type']
        onehot = [0.0] * len(GATE_TYPES)
        onehot[GATE_TO_IDX[gt]] = 1.0
        node_type_enc.append(onehot)

    # 如果没有边，添加自环边（避免 GCN 报错）
    if len(edges) == 0:
        edges = [(n, n) for n in node_names]
        print(f"WARNING: circuit {circuit_id} had no edges, added self-loops")

    # 构建边索引
    edge_index = []
    for u, v in edges:
        if u in node_names and v in node_names:
            edge_index.append([node_names.index(u), node_names.index(v)])

    # 转换为 PyG 格式 (2, E)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_type_enc = torch.tensor(node_type_enc, dtype=torch.float)

    return node_names, node_type_enc, edge_index