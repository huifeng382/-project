"""coverage_report_v2.json + sc_expansion.json 深检"""
import json, re, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("========== 1. coverage_report_v2.json ==========")
with open('data/coverage_report_v2.json', encoding='utf-8') as f:
    cov = json.load(f)

def walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")
    else:
        print(f"  {path} = {obj}")

walk(cov)
print(f"\n顶层 keys: {list(cov.keys())}")

# 检查每个 batch 的 fields 覆盖了哪些列、是否含 transistor 子字段
for bk, bv in cov.items():
    if isinstance(bv, dict) and 'fields' in bv:
        print(f"\n--- {bk}: total_rows={bv.get('total_rows')}, fields={len(bv['fields'])} 项 ---")
        for fn, fs in bv['fields'].items():
            print(f"    {fn}: {fs}")

print("\n========== 2. sc_expansion.json ==========")
with open('data/sc_expansion.json', encoding='utf-8') as f:
    sc = json.load(f)
print(f"总条目: {len(sc)}")

# 空/无效 subcircuit
empty = [k for k, v in sc.items() if not isinstance(v, dict) or not v.get('subcircuit')]
no_list = [k for k, v in sc.items() if isinstance(v, dict) and not isinstance(v.get('subcircuit'), list)]
empty_sub = [k for k, v in sc.items() if isinstance(v, dict) and isinstance(v.get('subcircuit'), list) and len(v['subcircuit']) == 0]
print(f"subcircuit 缺失/非列表: {len(no_list)}   空列表: {len(empty_sub)}   合计不可展开: {len(empty)} ({len(empty)/len(sc)*100:.1f}%)")

# 结构无效（缺 inst/cell/inputs/output）
bad_struct = 0; ref_cells = set(); n_inst = 0
for k, v in sc.items():
    if not isinstance(v, dict) or not isinstance(v.get('subcircuit'), list):
        continue
    for inst in v['subcircuit']:
        if not isinstance(inst, dict):
            bad_struct += 1; continue
        if not all(x in inst for x in ('inst', 'cell', 'inputs', 'output')):
            bad_struct += 1
        if isinstance(inst.get('cell'), str):
            ref_cells.add(inst['cell'])
        n_inst += 1
print(f"结构缺字段的实例: {bad_struct}/{n_inst}   引用的标准单元数: {len(ref_cells)}")

# 引用的 cell 是否都在 std_cells.lib
lib_cells = set()
if True:
    with open('data/std_cells.lib', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.match(r'\s*cell\s*\(\s*([^)]+?)\s*\)', line)
            if m:
                lib_cells.add(m.group(1).strip().strip('"'))
print(f"std_cells.lib 单元数: {len(lib_cells)}")
missing_lib = ref_cells - lib_cells
print(f"sc_expansion 引用但 lib 没有的单元: {len(missing_lib)} 个")
if missing_lib:
    print(f"  样例: {sorted(missing_lib)[:10]}")

# 空 subcircuit 的 key 形态（是否集中在某类名字）
print(f"\n不可展开的 {len(empty)} 个名字样例: {sorted(empty)[:15]}")
# 按前缀统计
from collections import Counter
pref = Counter(k.split('_')[1] if '_' in k else k for k in empty)
print(f"不可展开名字前缀分布(前10): {pref.most_common(10)}")

# V1 关键名字检查（旧训练数据用的）
print("\n--- V1 旧数据常用名（应保留且可展开）---")
for name in ['SC_INV', 'SC_INV_WIRE', 'SC_AND', 'SC_JOIN', 'SC_JOIN_OR_WIRE_AND_WIRE_AND_OR_WIRE_AND_WIRE_AND',
             'SC_JOIN_OR_OR', 'SC_JOIN_OR_WIRE_AND_AND_AND_AND_WIRE_WIRE_WIRE_WIRE_WIRE_OR_WIRE_AND_AND_AND_AND_WIRE_WIRE_WIRE_WIRE_WIRE']:
    v = sc.get(name)
    if v is None:
        print(f"  {name}: ❌ 不存在")
    elif isinstance(v, dict) and isinstance(v.get('subcircuit'), list) and len(v['subcircuit']) > 0:
        print(f"  {name}: ✓ 可展开 ({len(v['subcircuit'])} 实例)")
    else:
        print(f"  {name}: ❌ 存在但展开为空")

# V2 常用名检查
print("\n--- V2 数据常用名（cell_types 里出现最多的前10）---")
import pandas as pd
st = pd.read_parquet('data/batch_v2_full/circuit_static.parquet')
from collections import Counter as C2
ct_cnt = C2()
for c in st['cell_types_json']:
    arr = json.loads(c)
    for x in arr:
        ct_cnt[x] += 1
for name, cnt in ct_cnt.most_common(10):
    v = sc.get(name)
    if v is None:
        status = '❌ 不存在'
    elif isinstance(v, dict) and isinstance(v.get('subcircuit'), list) and len(v['subcircuit']) > 0:
        status = f'✓ 可展开 ({len(v["subcircuit"])} 实例)'
    else:
        status = '❌ 存在但展开为空'
    print(f"  {name} (x{cnt}): {status}")
