"""结构先验三列在 V3 上的实测分布（data_loader.py:715-719 的口径）。

`struct_prior = [transistor_count, #SC_AND, #SC_INV_WIRE]`，`model.py:50`
用 `nn.Linear(3, ...) + ReLU` **直接吃原始值**（没有 scaler，没有归一化）。

第 2/3 列用的是 V2 时代的**写死名字匹配**：
    sum(1 for g in ct if 'SC_AND' in str(g) and 'SC_AND_' not in str(g))
    sum(1 for g in ct if 'SC_INV_WIRE' in str(g))
若 V3 换了宏命名，这两列会对**所有**电路恒为 0 —— 静默，不报错，
模型只是永远学不到这两维（相当于 struct_prior 退化成 1 维）。

本脚本只读，不改文件、不跑训练。
"""
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq                                  # noqa: E402


def check(path, label):
    if not os.path.exists(path):
        print(f'[{label}] 无文件，跳过')
        return
    rows = pq.read_table(path, columns=['cell_types_json']).to_pylist()
    n = len(rows)
    z_and = z_inv = z_both = 0
    names = Counter()
    per_ct_len = []
    for r in rows:
        try:
            ct = json.loads(r['cell_types_json'])
        except Exception:
            ct = []
        if not isinstance(ct, list):
            ct = []
        per_ct_len.append(len(ct))
        for g in ct:
            names[str(g)] += 1
        a = sum(1 for g in ct if 'SC_AND' in str(g) and 'SC_AND_' not in str(g))
        b = sum(1 for g in ct if 'SC_INV_WIRE' in str(g))
        if a == 0:
            z_and += 1
        if b == 0:
            z_inv += 1
        if a == 0 and b == 0:
            z_both += 1
    print(f'[{label}] 电路 {n}  每组 cell_types 长度中位 '
          f'{sorted(per_ct_len)[n // 2]}')
    print(f'    #SC_AND == 0 的电路      : {z_and}/{n} = {z_and / n:.1%}')
    print(f'    #SC_INV_WIRE == 0 的电路 : {z_inv}/{n} = {z_inv / n:.1%}')
    print(f'    两列**同时**为 0 的电路   : {z_both}/{n} = {z_both / n:.1%}'
          f'   ← 这些电路的结构先验只剩 transistor_count 一维')
    print(f'    不同 cell 名总数 {len(names)}；最常见 12 个：')
    for nm, c in names.most_common(12):
        print(f'        {c:>7d}  {nm}')


print('=== V3 ===')
check(os.path.join(ROOT, 'data', 'v3_delivery', 'circuit_static.parquet'), 'V3')
print()
print('=== V2 batch_v2_full（对照，这两列是给它设计的）===')
check(os.path.join(ROOT, 'data', 'batch_v2_full', 'circuit_static.parquet'), 'V2 full')
