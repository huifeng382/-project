import pandas as pd
import json
import os

def convert_batch02():
    BATCH02_DIR = "data/batch_02"
    OUT_DIR = "data/batch_02_converted"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读取原始数据
    static_df = pd.read_parquet(f"{BATCH02_DIR}/circuit_static.parquet")
    dynamic_df = pd.read_parquet(f"{BATCH02_DIR}/timing_arcs.parquet")

    print("静态表列名:", static_df.columns.tolist())
    print("动态表列名:", dynamic_df.columns.tolist())

    # ---------- 转换静态表 ----------
    # 重命名电路ID列
    static_df = static_df.rename(columns={'candidate_id': 'circuit_id'})
    # 重命名网表列：如果有 gate_level_netlist_std 则重命名，并删除可能存在的旧 gate_level_netlist 列
    if 'gate_level_netlist_std' in static_df.columns:
        static_df = static_df.drop(columns=['gate_level_netlist'], errors='ignore')
        static_df = static_df.rename(columns={'gate_level_netlist_std': 'gate_level_netlist'})
    # 解析 pin_loads_json
    def parse_loads(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except:
                return {}
        elif isinstance(x, dict):
            return x
        else:
            return {}
    static_df['pin_loads_dict'] = static_df['pin_loads_json'].apply(parse_loads)
    
    # 保留必要列
    static_out = static_df[['circuit_id', 'gate_level_netlist']].copy()
    static_out.to_parquet(f"{OUT_DIR}/circuit_static.parquet", index=False)

    # ---------- 转换动态表 ----------
    # 重命名电路ID列
    dynamic_df = dynamic_df.rename(columns={'candidate_id': 'circuit_id'})
    # 确保 vector 长度为5
    dynamic_df['vector'] = dynamic_df['vector'].astype(str).str.zfill(5)
    
    # 补全缺少的 slew_e 和 arrival_e
    if 'slew_s' in dynamic_df.columns:
        dynamic_df['slew_e'] = dynamic_df['slew_s']
    else:
        dynamic_df['slew_e'] = 0.0
    if 'arrival_time_s' in dynamic_df.columns:
        dynamic_df['arrival_e'] = dynamic_df['arrival_time_s']
    else:
        dynamic_df['arrival_e'] = 0.0
    
    # 重命名已有的 arrival_time_a -> arrival_a 等
    for pin in ['a', 'b', 'c', 'd']:
        old = f'arrival_time_{pin}'
        new = f'arrival_{pin}'
        if old in dynamic_df.columns:
            dynamic_df = dynamic_df.rename(columns={old: new})
    
    # 添加负载列 load_a ~ load_e
    loads_map = static_df.set_index('circuit_id')['pin_loads_dict'].to_dict()
    for pin in ['a', 'b', 'c', 'd', 'e']:
        def get_load(row):
            loads = loads_map.get(row['circuit_id'], {})
            return loads.get(pin, 0.0)
        dynamic_df[f'load_{pin}'] = dynamic_df.apply(get_load, axis=1)
    
    # 确保 DELAY 列
    if 'DELAY' not in dynamic_df.columns:
        if 'delay_s' in dynamic_df.columns:
            dynamic_df = dynamic_df.rename(columns={'delay_s': 'DELAY'})
        else:
            raise KeyError("找不到 DELAY 列")
    
    # 保留 batch01 中存在的列
    required_cols = ['circuit_id', 'vector', 'direction', 'switching_pin',
                     'slew_a', 'slew_b', 'slew_c', 'slew_d', 'slew_e',
                     'arrival_a', 'arrival_b', 'arrival_c', 'arrival_d', 'arrival_e',
                     'load_a', 'load_b', 'load_c', 'load_d', 'load_e',
                     'DELAY']
    exist_cols = [col for col in required_cols if col in dynamic_df.columns]
    dynamic_df = dynamic_df[exist_cols].copy()
    dynamic_df = dynamic_df.fillna(0.0)
    
    dynamic_df.to_parquet(f"{OUT_DIR}/timing_arcs.parquet", index=False)

    print(f"转换完成！新数据保存在 {OUT_DIR}")
    print(f"静态表行数: {len(static_out)}")
    print(f"动态表行数: {len(dynamic_df)}")
    print(dynamic_df.head(2))

if __name__ == "__main__":
    convert_batch02()