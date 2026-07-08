"""
标准单元库 LIB 表解析和双线性插值查表。
使用 NLDM 7×7 表格，输入 slew → 负载 → 门延迟。
"""
import re, json, os
import numpy as np


def parse_lib(lib_path):
    """
    解析 Liberty .lib 文件，提取每个标准单元的 NLDM 查找表。
    返回: dict, key=cell_name, value=dict with 'rise'/'fall' tables
    """
    with open(lib_path, 'r') as f:
        content = f.read()

    cells = {}
    # 用 cell ( 分割，而非嵌套大括号的正则
    blocks = content.split('cell (')
    for block in blocks[1:]:  # 第一个是文件头，跳过
        # 提取 cell name
        name_end = block.index(')')
        cell_name = block[:name_end].strip()
        # 提取 cell body（大括号匹配）
        body_start = block.index('{', name_end) + 1
        depth = 1
        i = body_start
        while i < len(block) and depth > 0:
            if block[i] == '{': depth += 1
            elif block[i] == '}': depth -= 1
            i += 1
        body = block[body_start:i-1]
        cells[cell_name] = _parse_cell_body(body)

    return cells


def _parse_cell_body(body):
    """解析一个 cell 的 timing 表"""
    # 找到所有 timing() 块
    result = {}
    timing_pattern = re.compile(
        r'timing\s*\(\s*\).*?cell_(\w+)\s*\((\w+)\).*?'
        r'index_1\s*\(\s*"([^"]*)"\s*\).*?'
        r'index_2\s*\(\s*"([^"]*)"\s*\).*?'
        r'values\s*\(\s*(.*?)\)\s*;\s*\}', re.DOTALL)
    for tm in timing_pattern.finditer(body):
        sense = tm.group(1)  # 'rise' or 'fall'
        table_name = tm.group(2)
        idx1 = [float(x.strip()) for x in tm.group(3).split(',')]
        idx2 = [float(x.strip()) for x in tm.group(4).split(',')]
        vals_str = tm.group(5)
        # 清理：去掉引号、反斜杠，按逗号分割
        vals_str = vals_str.replace('\\', '').replace('\n', ' ').replace('"', '')
        values = [float(x.strip()) for x in vals_str.split(',') if x.strip()]

        n_rows = len(idx1)
        n_cols = len(idx2)
        if len(values) == n_rows * n_cols:
            table = np.array(values).reshape(n_rows, n_cols)
            result[sense] = {'idx1': np.array(idx1), 'idx2': np.array(idx2), 'table': table}

    return result


def load_mapping(mapping_path):
    """加载 SC_→ASAP7 映射表"""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def build_lib_tensors(lib_cells, mapping):
    """
    将 LIB 表转为 PyTorch tensor，支持可微双线性插值。
    返回: (gate_names, idx1_t, idx2_t, tables_t)
           gate_names: list of 27 gate type names
           idx1_t: (27, 7) slew index
           idx2_t: (27, 7) load index
           tables_t: (27, 2, 7, 7) delay tables [gate, rise/fall, slew_idx, load_idx]
    """
    import torch
    gate_list = sorted(set(v for v in mapping.values() if v is not None and v in lib_cells))
    N = len(gate_list)
    idx1_t = torch.zeros(N, 7)
    idx2_t = torch.zeros(N, 7)
    tables_t = torch.zeros(N, 2, 7, 7)  # 2 = rise, fall

    for gi, gname in enumerate(gate_list):
        cell = lib_cells[gname]
        for si, sense in enumerate(['rise', 'fall']):
            if sense in cell:
                t = cell[sense]
                idx1_t[gi] = torch.tensor(t['idx1'][:7], dtype=torch.float)
                idx2_t[gi] = torch.tensor(t['idx2'][:7], dtype=torch.float)
                tables_t[gi, si] = torch.tensor(t['table'][:7, :7], dtype=torch.float)

    return gate_list, idx1_t, idx2_t, tables_t


def lib_batch_lookup(sc_gate_names, slew_ps, load_ff, gate_list, idx1_t, idx2_t,
                      tables_t, mapping, default=10.0):
    """
    PyTorch 可微批量 LIB 查表。
    sc_gate_names: list of str, SC_ 门类型名
    slew_ps: (N,) tensor, 输入 slew (ps)
    load_ff: (N,) tensor, 输出负载 (fF)
    返回: (N,) tensor, 门延迟 (ps)
    """
    import torch
    N = len(sc_gate_names)
    delays = torch.full((N,), default, dtype=torch.float)

    for i in range(N):
        sc_name = sc_gate_names[i]
        if sc_name not in mapping or mapping[sc_name] is None:
            continue
        asap7 = mapping[sc_name]
        if asap7 not in gate_list:
            continue
        gi = gate_list.index(asap7)
        # 用 rise 表（简化：后续可扩展方向判断）
        table = tables_t[gi, 0]  # (7, 7)
        idx1 = idx1_t[gi]  # (7,)
        idx2 = idx2_t[gi]  # (7,)

        x = torch.clamp(slew_ps[i], idx1[0], idx1[-1])
        y = torch.clamp(load_ff[i], idx2[0], idx2[-1])

        # searchsorted 在 PyTorch 中不可微，用 clamp+linear 近似
        # 用 torch.where 计算插值权重
        xi = torch.searchsorted(idx1, x).clamp(1, 6)
        x0, x1 = idx1[xi-1], idx1[xi]
        wx = (x - x0) / (x1 - x0 + 1e-10)

        yi = torch.searchsorted(idx2, y).clamp(1, 6)
        y0, y1 = idx2[yi-1], idx2[yi]
        wy = (y - y0) / (y1 - y0 + 1e-10)

        # 双线性插值（可微）
        q00 = table[xi-1, yi-1]
        q10 = table[xi, yi-1]
        q01 = table[xi-1, yi]
        q11 = table[xi, yi]
        d = (q00 * (1-wx) * (1-wy) + q10 * wx * (1-wy) +
             q01 * (1-wx) * wy + q11 * wx * wy)
        delays[i] = d

    return delays


def lookup_delay(cell_name, input_slew_ps, output_load_ff, lib_cells, mapping,
                 sense='rise', default_delay=10.0):
    """
    查表获取门延迟。
    cell_name: SC_ 门类型名
    input_slew_ps: 输入 slew (ps)
    output_load_ff: 输出负载 (fF)
    lib_cells: parse_lib() 返回的结果
    mapping: load_mapping() 返回的结果
    sense: 'rise' 或 'fall'
    """
    if cell_name not in mapping:
        return default_delay
    asap7_name = mapping[cell_name]
    if asap7_name is None or asap7_name not in lib_cells:
        return default_delay

    cell_data = lib_cells[asap7_name]
    if sense not in cell_data:
        return default_delay

    t = cell_data[sense]
    idx1, idx2, table = t['idx1'], t['idx2'], t['table']

    # 双线性插值
    x = np.clip(input_slew_ps, idx1[0], idx1[-1])
    y = np.clip(output_load_ff, idx2[0], idx2[-1])

    i = np.searchsorted(idx1, x) - 1
    j = np.searchsorted(idx2, y) - 1
    i = max(0, min(i, len(idx1) - 2))
    j = max(0, min(j, len(idx2) - 2))

    x1, x2 = idx1[i], idx1[i+1]
    y1, y2 = idx2[j], idx2[j+1]
    q11, q12 = table[i, j], table[i, j+1]
    q21, q22 = table[i+1, j], table[i+1, j+1]

    return (q11 * (x2-x) * (y2-y) + q21 * (x-x1) * (y2-y) +
            q12 * (x2-x) * (y-y1) + q22 * (x-x1) * (y-y1)) / ((x2-x1) * (y2-y1))
