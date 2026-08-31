"""检查 batch_v2_m4 与旧数据的 expr 编号衔接(防切分冲突) + 数据完整性"""
import pandas as pd
import os

proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data = os.path.join(proj, 'data')

# 旧数据 expr 范围
for name in ['batch_v2_full', 'batch_v2_rest', 'batch_v2_io']:
    p = os.path.join(data, name)
    if os.path.isdir(p):
        fps = [os.path.join(p, f) for f in os.listdir(p)
               if f.startswith('circuit_static') and f.endswith('.parquet')]
        if fps:
            s = pd.read_parquet(fps[0])
            print(f'{name}: {len(s)} 电路, expr {s["expr"].min()} ~ {s["expr"].max()}')

m4 = pd.read_parquet(os.path.join(data, 'batch_v2_m4', 'circuit_static.parquet'))
print(f'batch_v2_m4: {len(m4)} 电路, expr {m4["expr"].min()} ~ {m4["expr"].max()}')

# 输入/输出引脚数分布(5 形状验证)
print('\n5 形状验证(入/出引脚数):')
m4['n_in'] = m4['input_pins_json'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
m4['n_out'] = m4['output_pins_json'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
print(m4.groupby(['n_in', 'n_out']).size().to_string())

# wave 完整性: 每电路是否都有 wave
ddf = pd.read_parquet(os.path.join(data, 'batch_v2_m4', 'timing_arcs.parquet'))
print(f'\ntiming_arcs: {len(ddf)} 行, 电路 {ddf["circuit_id"].nunique()}')
print('wave 缺失:', ddf['transistor_wave_json'].isna().sum())
print('DELAY 缺失:', ddf['DELAY'].isna().sum())
print('corner:', ddf['corner'].unique())
