"""
方案B：门类型分层抽样 — 将数据削减到 ~10万样本
- Batch 1: 130电路 × 30 corners × 1 vector/combo → ~31K
- Batch 2: 1,200电路 × 9 corners × all vectors → ~68K
- 合计: ~99K
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json
import numpy as np
import shutil

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_and_clean(name):
    """读取并清洗数据"""
    d = pd.read_parquet(os.path.join(DATA_DIR, name, "timing_arcs.parquet"))
    s = pd.read_parquet(os.path.join(DATA_DIR, name, "circuit_static.parquet"))

    # 列名规范化
    s = s.rename(columns={"candidate_id": "circuit_id"})
    d = d.rename(columns={"candidate_id": "circuit_id", "delay_s": "DELAY"})
    s["circuit_id"] = s["circuit_id"].astype(str)
    d["circuit_id"] = d["circuit_id"].astype(str)

    # 清洗负延迟
    before = len(d)
    d = d[d["DELAY"] > 0].copy()
    print(f"  {name}: removed {before - len(d)} rows with DELAY <= 0, {len(d)} remaining")

    return s, d


def get_cell_types_per_circuit(static_df):
    """返回 {circuit_id: set of cell types}"""
    result = {}
    for _, row in static_df.iterrows():
        cid = row["circuit_id"]
        try:
            types = json.loads(row["cell_types_json"]) if isinstance(row["cell_types_json"], str) else row["cell_types_json"]
            result[cid] = set(types)
        except Exception:
            result[cid] = set()
    return result


def greedy_coverage_select(circuit_cell_types, n_select, all_types=None):
    """
    贪心选择电路，最大化 cell type 覆盖率。
    优先覆盖稀有类型（出现次数少的）。
    """
    if all_types is None:
        all_types = set()
        for types in circuit_cell_types.values():
            all_types.update(types)

    # 按类型稀有度加权：稀有类型权重高
    type_freq = {}
    for types in circuit_cell_types.values():
        for t in types:
            type_freq[t] = type_freq.get(t, 0) + 1
    # 权重 = 1/freq（稀有类型权重高）
    type_weight = {t: 1.0 / f for t, f in type_freq.items()}

    selected = []
    covered = set()
    remaining = set(circuit_cell_types.keys())

    while len(selected) < n_select and remaining:
        # 选覆盖最多"新"类型的电路（按稀有度加权）
        best_cid = None
        best_score = -1
        for cid in remaining:
            new_types = circuit_cell_types[cid] - covered
            score = sum(type_weight.get(t, 1.0) for t in new_types)
            if score > best_score:
                best_score = score
                best_cid = cid

        if best_cid is None or best_score == 0:
            # 没有新类型了，随机补足
            rest = list(remaining)[: n_select - len(selected)]
            selected.extend(rest)
            break

        selected.append(best_cid)
        covered.update(circuit_cell_types[best_cid])
        remaining.remove(best_cid)

    coverage = len(covered) / len(all_types) * 100 if all_types else 0
    print(f"  Selected {len(selected)} circuits, coverage: {len(covered)}/{len(all_types)} "
          f"({coverage:.1f}%) cell types")
    return selected


def downsample_batch1(s, d, n_circuits=130):
    """Batch 1: 贪心选电路 + 每(corner,pin,dir)只留1个vector"""
    print("\n=== Batch 1: Handpicked ===")
    circuit_cell_types = get_cell_types_per_circuit(s)

    # 1. 贪心选择电路
    all_cids = sorted(d["circuit_id"].unique())
    valid_cids = [c for c in all_cids if c in circuit_cell_types]
    selected_cids = greedy_coverage_select(
        {c: circuit_cell_types[c] for c in valid_cids}, n_circuits
    )

    # 2. 过滤动态数据
    d_sel = d[d["circuit_id"].isin(selected_cids)].copy()

    # 3. 每个 (circuit, corner, switching_pin, direction) 只保留第1个vector
    group_cols = ["circuit_id", "corner", "switching_pin", "direction"]
    d_sel = d_sel.sort_values(["circuit_id", "corner", "switching_pin", "direction", "vector"])
    d_sel = d_sel.groupby(group_cols, as_index=False).first()

    print(f"  Samples: {len(d_sel):,} (target: ~31,000)")
    print(f"  Circuits: {d_sel['circuit_id'].nunique()}")

    # 过滤静态数据
    s_sel = s[s["circuit_id"].isin(selected_cids)].copy()
    return s_sel, d_sel


def downsample_batch2(s, d, n_circuits=1200):
    """Batch 2: 贪心选电路，保留全部样本"""
    print("\n=== Batch 2: E-graph ===")
    circuit_cell_types = get_cell_types_per_circuit(s)

    # 1. 贪心选择电路
    all_cids = sorted(d["circuit_id"].unique())
    valid_cids = [c for c in all_cids if c in circuit_cell_types]
    selected_cids = greedy_coverage_select(
        {c: circuit_cell_types[c] for c in valid_cids}, n_circuits
    )

    # 2. 过滤数据（保留全部样本）
    d_sel = d[d["circuit_id"].isin(selected_cids)].copy()
    s_sel = s[s["circuit_id"].isin(selected_cids)].copy()

    print(f"  Samples: {len(d_sel):,} (target: ~68,000)")
    print(f"  Circuits: {d_sel['circuit_id'].nunique()}")

    # 引脚分布
    pin_dist = d_sel["switching_pin"].value_counts(normalize=True)
    for p, v in pin_dist.items():
        print(f"    {p}: {v*100:.1f}%")

    return s_sel, d_sel


def save_dataset(s, d, out_dir):
    """保存为 parquet + metadata"""
    out_path = os.path.join(DATA_DIR, out_dir)
    os.makedirs(out_path, exist_ok=True)

    s.to_parquet(os.path.join(out_path, "circuit_static.parquet"), index=False)
    d.to_parquet(os.path.join(out_path, "timing_arcs.parquet"), index=False)

    meta = {
        "dataset_name": out_dir,
        "description": "Plan B downsampled: gate-type stratified selection",
        "n_circuits": int(d["circuit_id"].nunique()),
        "n_samples": len(d),
        "n_corners_per_circuit": int(d.groupby("circuit_id")["corner"].nunique().mean()),
        "seed": RANDOM_SEED,
        "parent_datasets": ["dataset_batch1_handpicked", "dataset_batch2_egraph_1k"],
    }
    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {out_path}/")
    print(f"    circuit_static.parquet: {len(s):,} circuits")
    print(f"    timing_arcs.parquet: {len(d):,} samples")


def main():
    print("=" * 60)
    print("  Plan B: Gate-Type Stratified Downsampling")
    print("  Target: ~100K total (30K batch1 + 70K batch2)")
    print("=" * 60)

    # ---- Batch 1 ----
    s1, d1 = load_and_clean("dataset_batch1_handpicked")
    s1_sel, d1_sel = downsample_batch1(s1, d1, n_circuits=130)
    save_dataset(s1_sel, d1_sel, "batch1_30k")

    # ---- Batch 2 ----
    s2, d2 = load_and_clean("dataset_batch2_egraph_1k")
    s2_sel, d2_sel = downsample_batch2(s2, d2, n_circuits=1200)
    save_dataset(s2_sel, d2_sel, "batch2_70k")

    total = len(d1_sel) + len(d2_sel)
    print(f"\n{'=' * 60}")
    print(f"  Total samples: {total:,} (target: ~100,000)")
    print(f"  Batch 1: {len(d1_sel):,} | Batch 2: {len(d2_sel):,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
