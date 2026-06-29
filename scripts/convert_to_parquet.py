import pandas as pd
import json
import os

# 设置路径（根据实际情况修改）
DATA_DIR = "data"
STATIC_JSON = os.path.join(DATA_DIR, "static_features.json")
DYNAMIC_CSV = os.path.join(DATA_DIR, "circuit_dataset.csv")

OUTPUT_STATIC = os.path.join(DATA_DIR, "circuit_static.parquet")
OUTPUT_DYNAMIC = os.path.join(DATA_DIR, "timing_arcs.parquet")

def convert_static():
    with open(STATIC_JSON, 'r') as f:
        static_data = json.load(f)
    
    records = []
    for cid, info in static_data.items():
        record = {
            "circuit_id": cid,
            "transistor_count": info.get("transistor_count", 0),
            "gate_level_netlist": info.get("gate_level_netlist", ""),
            "pin_loads": json.dumps(info.get("pin_loads", {})),  # 转为 JSON 字符串
            "output_load": info.get("output_load", 0.0),
        }
        # 可选：添加 input_pins 列表（如果需要）
        if "input_pins" in info:
            record["input_pins"] = json.dumps(info["input_pins"])
        records.append(record)
    
    df_static = pd.DataFrame(records)
    df_static.to_parquet(OUTPUT_STATIC, index=False)
    print(f"Saved static data to {OUTPUT_STATIC}, rows: {len(df_static)}")

def convert_dynamic():
    df = pd.read_csv(DYNAMIC_CSV)
    # 可选：确保列名统一，例如 'candidate' 重命名为 'circuit_id'
    if 'candidate' in df.columns:
        df.rename(columns={'candidate': 'circuit_id'}, inplace=True)
    # 确保 DELAY 列存在
    if 'DELAY' not in df.columns:
        raise KeyError("DELAY column not found in CSV")
    df.to_parquet(OUTPUT_DYNAMIC, index=False)
    print(f"Saved dynamic data to {OUTPUT_DYNAMIC}, rows: {len(df)}")

if __name__ == "__main__":
    convert_static()
    convert_dynamic()
    print("Conversion completed.")