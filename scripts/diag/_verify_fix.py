"""修复版遗留问题核实：ids_charge / gate_states 大小写检查 / sc_expansion 12空 / parasitic_caps 12个"""
import json, sys
import pandas as pd
from collections import Counter
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

st = pd.read_parquet('data/batch_v2_full/circuit_static.parquet')
dy = pd.read_parquet('data/batch_v2_full/timing_arcs.parquet')
st['circuit_id'] = st['circuit_id'].astype(str)
dy['circuit_id'] = dy['circuit_id'].astype(str)

def lj(x):
    return json.loads(x) if isinstance(x, str) else x

print("===== 1. ids_charge vs ids_avg（修复版）=====")
n = 0; eq = 0; ratio = []
for _, r in dy.sample(3000).iterrows():
    tw = lj(r['transistor_wave_json'])
    if not isinstance(tw, dict): continue
    for m, sub in tw.items():
        if isinstance(sub, dict) and 'ids_avg' in sub and 'ids_charge' in sub:
            a, c = sub['ids_avg'], sub['ids_charge']
            if isinstance(a, (int, float)) and isinstance(c, (int, float)):
                n += 1
                if abs(c - a) < 1e-9:
                    eq += 1
                if a > 0:
                    ratio.append(c / a)
print(f"  晶体管 {n} 个, ids_charge==ids_avg(1e-9): {eq} ({eq/n*100:.1f}%)")
import numpy as np
ratio = np.array(ratio)
print(f"  ids_charge/ids_avg: 中位 {np.median(ratio):.3f}  完全=1 占比 {np.mean(ratio==1)*100:.1f}%")
# 样例
for _, r in dy.head(3).iterrows():
    tw = lj(r['transistor_wave_json'])
    if isinstance(tw, dict) and tw:
        m0 = list(tw)[0]
        print(f"  样例 {r['circuit_id']} {m0}: ids_avg={tw[m0]['ids_avg']} ids_peak={tw[m0]['ids_peak']} ids_rise_time={tw[m0]['ids_rise_time']} ids_charge={tw[m0]['ids_charge']}")

print("\n===== 2. gate_states 大小写（从动态表核）=====")
nlmap = dict(zip(st['circuit_id'], st['gate_level_netlist']))
ok = 0; n2 = 0; sample_keys = None
for _, r in dy.sample(3000).iterrows():
    gs = lj(r['gate_states_json'])
    gates = {}
    for line in (nlmap.get(r['circuit_id'], '') or '').splitlines():
        line = line.strip()
        if line.startswith('X_'):
            t = line.split()
            if len(t) >= 2:
                gates[t[0]] = t[-1]
    if not isinstance(gs, dict) or not gates:
        continue
    n2 += 1
    if set(gs.keys()) == set(gates.keys()):
        ok += 1
    if sample_keys is None:
        sample_keys = (sorted(gs.keys())[:3], sorted(gates.keys())[:3])
print(f"  gate_states key == 网表实例名: {ok}/{n2}")
print(f"  样例: gs keys={sample_keys[0]}  netlist keys={sample_keys[1]}")

print("\n===== 3. sc_expansion 剩 12 个空的名字 =====")
sc = json.load(open('data/sc_expansion.json', encoding='utf-8'))
cells = set()
for c in st['cell_types_json']:
    for x in lj(c):
        cells.add(x)
sc_cells = {c for c in cells if c.startswith('SC_')}
bad = {c for c in sc_cells if not isinstance(sc.get(c), dict) or not sc.get(c, {}).get('subcircuit')}
print(f"  V2 用到的 SC_ 名 {len(sc_cells)} 个, 不可展开 {len(bad)} 个:")
for c in sorted(bad):
    v = sc.get(c)
    print(f"    {c}: value={'None' if v is None else type(v).__name__}")
print(f"  sc_expansion 全文件 null 数: {sum(1 for v in sc.values() if v is None)}")

print("\n===== 4. parasitic_caps 子字段失败的电路 =====")
pc_bad = []
for _, r in st.sample(3000).iterrows():
    pc = lj(r['parasitic_caps_json'])
    if not isinstance(pc, dict):
        pc_bad.append((r['circuit_id'], 'not dict'))
        continue
    for inst, sub in pc.items():
        if not isinstance(sub, dict) or 'out' not in sub or not any(k.startswith('in_') for k in sub):
            pc_bad.append((r['circuit_id'], inst, sub if isinstance(sub, dict) else type(sub).__name__))
            break
print(f"  异常电路数: {len(pc_bad)}")
for x in pc_bad[:10]:
    print("   ", x)
