"""生成 serve 测试候选：从 batch_v2_io 取前 N 个电路（含真 avg_delay），写 candidates JSON。
用法: python scripts/diag/_gen_candidates.py --n 8 --out C:/Users/24169/models/candidates_test.json
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--out', default=os.path.join('models', 'candidates_test.json'))
args = ap.parse_args()

st = pd.read_parquet('data/batch_v2_io/circuit_static.parquet')
dy = pd.read_parquet('data/batch_v2_io/timing_arcs.parquet')
dy['circuit_id'] = dy['circuit_id'].astype(str)
cands = []
for _, r in st.head(args.n).iterrows():
    cid = str(r['circuit_id'])
    true_avg = float(dy[dy['circuit_id'] == cid]['DELAY'].mean())
    cands.append({'id': cid, 'netlist': r['gate_level_netlist'],
                  'input_pins': json.loads(r['input_pins_json']),
                  'output_pins': json.loads(r['output_pins_json']),
                  'true_avg_delay': true_avg})
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(cands, f, ensure_ascii=False)
print(f"wrote {len(cands)} candidates -> {args.out}")
for c in cands:
    print(f"  {c['id']}: in={len(c['input_pins'])} out={len(c['output_pins'])} "
          f"true_avg={c['true_avg_delay']*1e12:.1f}ps")
