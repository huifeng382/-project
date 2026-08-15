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

# ---- 结构特征：固定逻辑类别替代 638 类 cell 名嵌入（通用、永不 OOV）----
LOGIC_TYPES = ['INV', 'NAND', 'NOR', 'AND', 'OR', 'AOI', 'OAI', 'XOR', 'BUF', 'COMPLEX']
LOGIC_TO_IDX = {t: i for i, t in enumerate(LOGIC_TYPES)}

import os as _os
import json as _json

_SC_EXPANSION = None


def _load_sc_expansion():
    global _SC_EXPANSION
    if _SC_EXPANSION is None:
        _p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                           'data', 'sc_expansion.json')
        try:
            _SC_EXPANSION = _json.load(open(_p, encoding='utf-8'))
        except Exception:
            _SC_EXPANSION = {}
    return _SC_EXPANSION


def _asap7_cell_feat(cell):
    """ASAP7 单元名 -> (逻辑函数, 输入数, 驱动, 晶体管数)。NAND2x1 -> ('NAND',2,1,4)。"""
    m = re.match(r'^([A-Z]+)(\d*)x(\d+)', cell)
    if not m:
        return None
    func, nin_s, drive_s = m.group(1), m.group(2), m.group(3)
    nin = int(nin_s) if nin_s else 1
    drive = int(drive_s)
    if func == 'INV':
        n_t = 2
    elif func == 'BUF':
        n_t = 4
    elif func in ('NAND', 'NOR'):
        n_t = 2 * nin
    elif func in ('AND', 'OR'):
        n_t = 2 * nin + 2
    elif func in ('XOR', 'XNOR'):
        n_t = 12
    else:
        n_t = 2 * max(nin, 1)
    return func, nin, drive, n_t


def _compose_logic(funcs):
    if not funcs:
        return 'COMPLEX'
    if len(funcs) == 1:
        f = funcs[0]
        return {'INV': 'INV', 'BUF': 'BUF', 'NAND': 'NAND', 'NOR': 'NOR',
                'XOR': 'XOR', 'XNOR': 'XOR'}.get(f, 'COMPLEX')
    if len(funcs) == 2 and funcs[-1] in ('INV', 'BUF'):
        if funcs[0] == 'NAND':
            return 'AND'
        if funcs[0] == 'NOR':
            return 'OR'
        if funcs[0] == 'INV':
            return 'BUF'
    return 'COMPLEX'


def _name_fallback_logic(name):
    gt = name.upper()
    # SC_JOIN_*/SC_BRIDGE_*/SC_*_WIRE_* 都是复杂门，归 COMPLEX
    if 'JOIN' in gt or 'BRIDGE' in gt or 'WIRE' in gt:
        return 'COMPLEX'
    if 'NAND' in gt:
        return 'NAND'
    if 'NOR' in gt:
        return 'NOR'
    if 'XOR' in gt or 'XNOR' in gt:
        return 'XOR'
    if 'INV' in gt:
        return 'INV'
    if 'BUF' in gt:
        return 'BUF'
    if 'AND' in gt:
        return 'AND'
    if 'OR' in gt:
        return 'OR'
    return 'COMPLEX'


def gate_struct(name):
    """cell 名 -> (logic_type, n_transistors, drive, stack_height)。
    优先用 sc_expansion.json 的 ASAP7 展开；查不到回退名字关键字。"""
    exp = _load_sc_expansion().get(name)
    if isinstance(exp, dict):
        sub = exp.get('subcircuit')
        if sub:
            feats = [f for f in (_asap7_cell_feat(x.get('cell', '')) for x in sub) if f is not None]
            if feats:
                logic = _compose_logic([f[0] for f in feats])
                n_t = float(sum(f[3] for f in feats))
                drive = float(feats[-1][2])
                stack = float(max(f[1] for f in feats))
                return logic, n_t, drive, stack
    return _name_fallback_logic(name), 6.0, 1.0, 1.0


def rebuild_gate_types(cell_types):
    """
    从实际数据中动态构建门类型映射。
    cell_types: 所有出现的 cell 类型字符串的可迭代集合。
    调用后 GATE_TYPES 和 GATE_TO_IDX 会被重建，包含所有 found types +
    INPUT_PIN / OUTPUT_PIN / UNKNOWN_GATE。
    """
    global GATE_TYPES, GATE_TO_IDX
    reserved = ['INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE']
    # 用固定逻辑类别（10 类）替代动态 cell 名词表；cell_types 参数保留（兼容调用）但不再用于构建词表
    GATE_TYPES = LOGIC_TYPES + reserved
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
    
    # 节点门类型索引：用固定逻辑类别（10 类）替代 638 类 cell 名，另算晶体管数特征
    node_type_idx = []
    n_transistor_feat = []
    for n in node_names:
        gt = nodes[n]['type']
        if gt in ('INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE'):
            idx = GATE_TO_IDX.get(gt, GATE_TO_IDX['UNKNOWN_GATE'])
            n_t = 0.0
        else:
            logic, n_t, _drive, _stack = gate_struct(gt)
            idx = GATE_TO_IDX.get(logic, GATE_TO_IDX['UNKNOWN_GATE'])
        node_type_idx.append([float(idx)])
        n_transistor_feat.append(n_t)
    node_type_idx = torch.tensor(node_type_idx, dtype=torch.float)
    n_transistor_feat = torch.tensor([[x] for x in n_transistor_feat], dtype=torch.float)
    
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

    # 合并静态特征：门类型索引 + 扇出 + 深度 + 驱动 + 寄生延迟 + 逻辑努力 + 电努力 + 晶体管数
    node_static = torch.cat([node_type_idx, fanout_feat, depth_feat, drive_feat,
                              p_feat, g_feat, h_feat, n_transistor_feat], dim=1)

    return node_names, node_static, edge_index