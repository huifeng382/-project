"""V2 数据补充量化诊断"""
import pandas as pd, json, numpy as np

dy = pd.read_parquet('data/batch_v2_full/timing_arcs.parquet')
dy['circuit_id'] = dy['circuit_id'].astype(str)

def lj(x):
    return json.loads(x) if isinstance(x, str) else x

# 1) ids_charge vs ids_avg
n = 0; rel = []; 
for _, r in dy.sample(3000).iterrows():
    tw = lj(r['transistor_wave_json'])
    if not isinstance(tw, dict): continue
    for m, sub in tw.items():
        if isinstance(sub, dict) and 'ids_avg' in sub and 'ids_charge' in sub:
            a, c = sub['ids_avg'], sub['ids_charge']
            if isinstance(a, (int, float)) and isinstance(c, (int, float)) and a > 0:
                rel.append(abs(c - a) / a); n += 1
rel = np.array(rel)
print(f"ids_charge vs ids_avg: n={n}  median|Δ|/avg={np.median(rel)*100:.2f}%  "
      f"pct <1%差异={np.mean(rel < 0.01)*100:.1f}%  pct 完全相等={np.mean(rel == 0)*100:.1f}%")
print(f"  ids_charge 与 ids_avg 中位比值: {np.median([(json.loads(x) if isinstance(x,str) else x) for x in []]) if False else ''}")
# 直接比值
ratio = []
for _, r in dy.sample(3000).iterrows():
    tw = lj(r['transistor_wave_json'])
    if not isinstance(tw, dict): continue
    for m, sub in tw.items():
        if isinstance(sub, dict) and sub.get('ids_avg', 0) and sub.get('ids_charge', 0):
            ratio.append(sub['ids_charge'] / sub['ids_avg'])
ratio = np.array(ratio)
print(f"  ids_charge/ids_avg: median={np.median(ratio):.4f}  完全=1 的占比 {np.mean(ratio==1)*100:.1f}%")

# 2) supply_noise 全零占比
z = 0; n2 = 0
for _, r in dy.sample(5000).iterrows():
    sn = lj(r['supply_noise_json'])
    if isinstance(sn, dict):
        n2 += 1
        if sn.get('vdd_droop_mV') == 0 and sn.get('gnd_bounce_mV') == 0:
            z += 1
print(f"supply_noise 全零占比: {z}/{n2} = {z/n2*100:.1f}%")

# 3) gate_states 取值分布
vals = []
for _, r in dy.sample(2000).iterrows():
    gs = lj(r['gate_states_json'])
    if isinstance(gs, dict):
        vals += list(gs.values())
print(f"gate_states 值分布: 0={sum(1 for v in vals if v==0)}, 1={sum(1 for v in vals if v==1)}, 其它={len(vals)-sum(1 for v in vals if v in (0,1))}")

# 4) 行数异常电路详情
rc = dy.groupby('circuit_id').size()
print(f"\n每电路行数分布: {rc.value_counts().to_dict()}")
# 行数 ≠ 8 的电路
c_bad = rc[rc != 8].index
sub = dy[dy['circuit_id'].isin(c_bad[:5])]
print(f"\n非8行电路 {len(c_bad)} 个 (样例 {min(5,len(c_bad))}):")
for cid, g in sub.groupby('circuit_id'):
    print(f"  {cid}: rows={len(g)}")
    print("   (circuit,pin,dir) -> vectors:", g.groupby(['switching_pin','direction'])['vector'].apply(list).to_dict())

# 5) 重复行样例
dup = dy[dy.duplicated(subset=['circuit_id','corner','switching_pin','direction','vector'], keep=False)]
print(f"\n重复行: {len(dup)} 行, 涉及电路 {dup['circuit_id'].nunique()}")
print(dup.head(4).to_string(index=False))
