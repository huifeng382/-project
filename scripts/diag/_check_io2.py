"""batch_v2_io 细查：I/O 形状覆盖 / ids_charge 残留性质 / 组大小分布"""
import json, sys
import pandas as pd, numpy as np
from collections import Counter
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

st = pd.read_parquet('data/batch_v2_io/circuit_static.parquet')
dy = pd.read_parquet('data/batch_v2_io/timing_arcs.parquet')
st['circuit_id'] = st['circuit_id'].astype(str)
dy['circuit_id'] = dy['circuit_id'].astype(str)
def lj(x): return json.loads(x) if isinstance(x, str) else x

print("===== 1. I/O 形状分布（batch_v2_io, 4248 电路）=====")
n_in = st['input_pins_json'].map(lambda x: len(lj(x)))
n_out = st['output_pins_json'].map(lambda x: len(lj(x)))
shapes = Counter(zip(n_in, n_out))
print(f"不同形状数: {len(shapes)}")
for (ni, no), v in sorted(shapes.items(), key=lambda x: (x[0][1], x[0][0])):
    mark = ' ✅' if (ni, no) in [(4, 1)] else ''
    print(f"  {ni:>2} 入 / {no} 出 : {v:>4} 个{mark}")
# 与 benchmark 18 种形状对比
bench = {(1,1),(2,1),(3,1),(4,1),(5,1),(8,1),(9,1),(16,1),(2,2),(3,2),(5,2),(2,3),(4,3),(8,3),(7,4),(8,4),(5,5),(9,6)}
missing = bench - set(shapes.keys())
print(f"benchmark 18 形状中缺失: {sorted(missing) if missing else '无 —— 全部覆盖 ✅'}")

print("\n===== 2. ids_charge 残留性质 =====")
flag = 0; coinc = 0; copy_real = 0; n_active = 0
for _, r in dy.sample(5000).iterrows():
    tw = lj(r['transistor_wave_json'])
    if not isinstance(tw, dict): continue
    for m, sub in tw.items():
        if not isinstance(sub, dict) or 'ids_charge' not in sub: continue
        a, c, pk, rt = sub['ids_avg'], sub['ids_charge'], sub['ids_peak'], sub['ids_rise_time']
        if a == 0 and c == 0: continue
        n_active += 1
        if abs(c - a) < 1e-9:
            flag += 1
            if abs(c - round(pk * rt / 1000, 2)) < 1e-9:
                coinc += 1
            else:
                copy_real += 1
print(f"激活管 {n_active}: ==ids_avg {flag} ({flag/n_active*100:.2f}%), 其中取整巧合 {coinc}, 真复制残留 {copy_real}")

print("\n===== 3. 组大小分布 =====")
grp = st.groupby('expr')['circuit_id'].nunique()
print(f"组数 {len(grp)}, 中位 {grp.median():.0f}, 范围 [{grp.min()},{grp.max()}]")
print(f"  ≥10 变体: {(grp>=10).mean()*100:.1f}%")
print(f"  <10 变体: {(grp<10).sum()} 组 ({(grp<10).mean()*100:.1f}%)")
print(f"  =1 变体(退化组): {(grp==1).sum()} 组")
print(f"  组大小分布(前12): {grp.value_counts().sort_index().head(12).to_dict()}")

print("\n===== 4. 分桶比例 =====")
bins = pd.cut(n_in, [0, 2, 4, 8, 16], labels=['1~2', '3~4', '5~8', '9~16'])
print(f"输入分桶: {bins.value_counts(normalize=True).round(3).to_dict()}")
print(f"多输出占比: {(n_out>=2).mean()*100:.1f}%")
