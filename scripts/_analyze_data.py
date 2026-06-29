"""分析两个新数据集的结构，为削减到10w样本提供依据"""
import pandas as pd
import json
import numpy as np

for name in ['dataset_batch1_handpicked', 'dataset_batch2_egraph_1k']:
    d = pd.read_parquet(f'data/{name}/timing_arcs.parquet')
    s = pd.read_parquet(f'data/{name}/circuit_static.parquet')

    # 列名规范化
    d = d.rename(columns={'candidate_id': 'circuit_id', 'delay_s': 'DELAY'})
    d = d[d['DELAY'] > 0]
    d['circuit_id'] = d['circuit_id'].astype(str)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  总样本数: {len(d):,}")
    print(f"  电路数:   {d['circuit_id'].nunique()}")
    print(f"  corners:  {sorted(d['corner'].unique())}")
    print(f"  directions: {d['direction'].unique()}")
    print(f"  switching_pins: {sorted(d['switching_pin'].dropna().unique())}")

    # 每电路的样本数分布
    per_circuit = d.groupby('circuit_id').size()
    print(f"\n  每电路样本数: min={per_circuit.min()}, median={per_circuit.median():.0f}, "
          f"mean={per_circuit.mean():.0f}, max={per_circuit.max()}")

    # 每电路的 unique corners 数
    corners_per_circuit = d.groupby('circuit_id')['corner'].nunique()
    print(f"  每电路 corner 数: min={corners_per_circuit.min()}, max={corners_per_circuit.max()}, "
          f"mean={corners_per_circuit.mean():.1f}")

    # 每电路 unique vectors 数
    if 'vector' in d.columns:
        vecs_per_circuit = d.groupby('circuit_id')['vector'].nunique()
        print(f"  每电路 vector 数: min={vecs_per_circuit.min()}, max={vecs_per_circuit.max()}, "
              f"mean={vecs_per_circuit.mean():.1f}")

    # 每个 (circuit, corner, switching_pin, direction) 组合的样本数
    group_cols = ['circuit_id', 'corner', 'switching_pin', 'direction']
    combo_size = d.groupby(group_cols).size()
    print(f"\n  每个(circuit,corner,pin,dir)组合样本数:")
    print(f"    min={combo_size.min()}, median={combo_size.median():.0f}, "
          f"mean={combo_size.mean():.1f}, max={combo_size.max()}")
    multi = (combo_size > 1).sum()
    print(f"    有{multi}/{len(combo_size)}个组合有多于1个样本 (多个vector)")

    # 引脚多样性
    pins_per_circuit = d.groupby('circuit_id')['switching_pin'].nunique()
    print(f"\n  每电路引脚数: min={pins_per_circuit.min()}, median={pins_per_circuit.median():.0f}, "
          f"max={pins_per_circuit.max()}")

    # 引脚分布
    print(f"\n  引脚频率分布:")
    for pin, cnt in d['switching_pin'].value_counts().items():
        print(f"    {pin}: {cnt:>8,} ({cnt/len(d)*100:.1f}%)")

    # 门类型多样性 (从static)
    if 'cell_types_json' in s.columns:
        all_types = set()
        for types_json in s['cell_types_json']:
            types = json.loads(types_json) if isinstance(types_json, str) else types_json
            all_types.update(types)
        print(f"\n  独特 cell 类型数: {len(all_types)}")

    # vector 分布
    if 'vector' in d.columns:
        print(f"\n  vector 分布 (top 10):")
        for v, cnt in d['vector'].value_counts().head(10).items():
            print(f"    {v}: {cnt:>8,}")

print(f"\n{'='*60}")
print("  汇总")
print(f"{'='*60}")
total_clean = 0
for name in ['dataset_batch1_handpicked', 'dataset_batch2_egraph_1k']:
    d = pd.read_parquet(f'data/{name}/timing_arcs.parquet')
    d = d.rename(columns={'candidate_id': 'circuit_id', 'delay_s': 'DELAY'})
    clean = (d['DELAY'] > 0).sum()
    total_clean += clean
    print(f"  {name}: {clean:,} (过滤后)")
print(f"  总计: {total_clean:,}")
