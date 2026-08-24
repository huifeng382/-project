"""V2 数据细节检查：分开看 batch_v2 / batch_v2_full，定位合规问题根因。"""
import pandas as pd, json, sys

pd.set_option('display.max_colwidth', 80)

for d in ['data/batch_v2', 'data/batch_v2_full']:
    print(f"\n{'='*70}\n### {d}\n{'='*70}")
    st = pd.read_parquet(f'{d}/circuit_static.parquet')
    dy = pd.read_parquet(f'{d}/timing_arcs.parquet')
    st['circuit_id'] = st['circuit_id'].astype(str)
    dy['circuit_id'] = dy['circuit_id'].astype(str)
    print(f"static {len(st)} 行 | dynamic {len(dy)} 行 | 电路 {st['circuit_id'].nunique()}")

    print("\n-- static 列 --")
    print(list(st.columns))
    print("\n-- input_pins_json 样例(前8) --")
    for v in st['input_pins_json'].head(8):
        print("  ", v)
    print("\n-- output_pins_json 样例(前8) --")
    for v in st['output_pins_json'].head(8):
        print("  ", v)
    print("\n-- cell_types_json 样例(前5) --")
    for v in st['cell_types_json'].head(5):
        print("  ", v[:200])
    print("\n-- gate_level_netlist 前3个电路 --")
    for v in st['gate_level_netlist'].head(3):
        print("  " + v.replace('\n', '\n  ')[:500])
    print("\n-- parasitic_caps_json 样例(前3) --")
    for v in st['parasitic_caps_json'].head(3):
        s = str(v)
        print("  ", s[:200])
    print("\n-- transistor_count 分布 --")
    print("  ", st['transistor_count'].value_counts().head(8).to_dict())

    print("\n-- dynamic 列 --")
    print(list(dy.columns))
    print("\n-- corner 分布 --")
    print("  ", dy['corner'].value_counts().to_dict())
    print("\n-- direction 分布 --")
    print("  ", dy['direction'].value_counts().to_dict())
    print("\n-- switching_pin 分布(前10) --")
    print("  ", dy['switching_pin'].value_counts().head(10).to_dict())
    print("\n-- vector 样例(前8) --")
    for v in dy['vector'].head(8):
        print("  ", repr(v))
    print("\n-- slew_s / output_load_f 统计 --")
    print(f"  slew_s: min={dy['slew_s'].min():.3e} max={dy['slew_s'].max():.3e} nunique={dy['slew_s'].nunique()}")
    print(f"  output_load_f: nunique={dy['output_load_f'].nunique()} 值={dy['output_load_f'].unique()[:5]}")
    print("\n-- pin_slew_json 样例(前3) --")
    for v in dy['pin_slew_json'].head(3):
        print("  ", str(v)[:150])
    print("\n-- pin_load_json 样例(前3) --")
    for v in dy['pin_load_json'].head(3):
        print("  ", str(v)[:150])
    print("\n-- gate_states_json 样例(前3) --")
    for v in dy['gate_states_json'].head(3):
        print("  ", str(v)[:150])
    print("\n-- transistor_wave_json 样例(前2) --")
    for v in dy['transistor_wave_json'].head(2):
        print("  ", str(v)[:300])
    print("\n-- supply_noise_json 样例(前3) --")
    for v in dy['supply_noise_json'].head(3):
        print("  ", str(v)[:150])

# 两批重叠
s1 = set(pd.read_parquet('data/batch_v2/circuit_static.parquet', columns=['circuit_id'])['circuit_id'].astype(str))
s2 = set(pd.read_parquet('data/batch_v2_full/circuit_static.parquet', columns=['circuit_id'])['circuit_id'].astype(str))
print(f"\n重叠 circuit_id: {len(s1 & s2)}  (batch_v2 {len(s1)}, batch_v2_full {len(s2)})")
