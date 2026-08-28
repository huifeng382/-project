"""检查默认训练数据 batch_v2_full + batch_v2_rest 的合规性（2026-08-28 定：默认这两组）。

覆盖：行数/列/schema一致性/expr与电路重叠/组大小/行公式/DELAY/I-O分布/缺失值。
用法：python scripts/diag/_check_v3_data.py
"""
import pandas as pd, glob, json, os

REQ_STATIC = ['circuit_id', 'expr', 'candidate_idx', 'transistor_count', 'gate_level_netlist',
              'cell_types_json', 'input_pins_json', 'output_pins_json', 'pin_loads_json',
              'parasitic_caps_json']
REQ_DYN = ['circuit_id', 'expr', 'switching_pin', 'direction', 'DELAY', 'vector',
           'slew_s', 'output_load_f']


def lj(x):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except Exception:
        return None


def main():
    batches = ['batch_v2_full', 'batch_v2_rest']
    stats = {}
    for batch in batches:
        sp = sorted(glob.glob(f'data/{batch}/circuit_static*.parquet'))
        dp = sorted(glob.glob(f'data/{batch}/timing_arcs*.parquet'))
        st = pd.concat([pd.read_parquet(f) for f in sp], ignore_index=True)
        dyn = pd.concat([pd.read_parquet(f) for f in dp], ignore_index=True)
        st['circuit_id'] = st['circuit_id'].astype(str)
        dyn['circuit_id'] = dyn['circuit_id'].astype(str)
        stats[batch] = (st, dyn)
        print(f"\n=== {batch} ===")
        print(f"  电路={st['circuit_id'].nunique()}  动态行={len(dyn)}  expr={st['expr'].nunique()}")
        miss_s = [c for c in REQ_STATIC if c not in st.columns]
        miss_d = [c for c in REQ_DYN if c not in dyn.columns]
        print(f"  缺列: 静态{miss_s or '无'}  动态{miss_d or '无'}")
        d = dyn['DELAY'].dropna()
        print(f"  DELAY: {d.min():.2e}~{d.max():.2e}  NaN={int(dyn['DELAY'].isna().sum())}  "
              f"<=0: {int((dyn['DELAY'] <= 0).sum())}")
        # 重复行
        dup = int(dyn.duplicated(subset=dyn.columns).sum())
        print(f"  动态重复行: {dup}")
        # 组大小
        g = st.groupby('expr')['circuit_id'].nunique()
        print(f"  组大小: 中位={g.median():.0f} 单变体={(g==1).sum()} <10={(g<10).sum()}/{len(g)}")
        # 行公式
        rows_c = dyn.groupby('circuit_id').size()
        st2 = st.set_index('circuit_id')
        exp = (st2['input_pins_json'].map(lambda x: len(lj(x) or []))
               * st2['output_pins_json'].map(lambda x: len(lj(x) or [])) * 2)
        bad = int((rows_c.reindex(exp.index) != exp).sum())
        print(f"  行数!=2*N_in*M: {bad}")

    # 跨批一致性
    print("\n=== 跨批检查 ===")
    st_full, _ = stats['batch_v2_full']
    st_rest, _ = stats['batch_v2_rest']
    exp_overlap = set(st_full['expr'].astype(str)) & set(st_rest['expr'].astype(str))
    cid_overlap = set(st_full['circuit_id']) & set(st_rest['circuit_id'])
    print(f"  expr 重叠: {len(exp_overlap)}")
    print(f"  circuit_id 重叠: {len(cid_overlap)}")
    cols_full = set(st_full.columns); cols_rest = set(st_rest.columns)
    print(f"  schema 差异: 静态 full-only={cols_full - cols_rest or '无'} rest-only={cols_rest - cols_full or '无'}")
    # 合并后总览
    st = pd.concat([st_full, st_rest], ignore_index=True).drop_duplicates('circuit_id')
    n_in = st['input_pins_json'].map(lambda x: len(lj(x) or []))
    n_out = st['output_pins_json'].map(lambda x: len(lj(x) or []))
    print(f"  合并: 电路={len(st)} expr={st['expr'].nunique()}")
    print(f"  输入分桶: 1~2={((n_in<=2).mean()*100):.1f}% 3~4={(((n_in>=3)&(n_in<=4)).mean()*100):.1f}% "
          f"5~8={(((n_in>=5)&(n_in<=8)).mean()*100):.1f}% 9~16={((n_in>=9).mean()*100):.1f}%  "
          f"多输出={((n_out>=2).mean()*100):.1f}% M>=4={(n_out>=4).sum()}")

    # Rust benchmark 形状覆盖视图（46 个 .tl vs 当前数据）
    print("\n=== Rust benchmark 形状覆盖（46 个 .tl vs 当前数据）===")
    rust = {}
    for f in glob.glob('NetlistOpt/testbench/tl_cells/**/*.tl', recursive=True):
        t = open(f, encoding='utf-8', errors='ignore').read()
        try:
            ni = len([l for l in t.splitlines() if l.startswith('INORDER')][0]
                     .replace('INORDER =', '').replace(';', '').split())
            no = len([l for l in t.splitlines() if l.startswith('OUTORDER')][0]
                     .replace('OUTORDER =', '').replace(';', '').split())
        except Exception:
            continue
        rust.setdefault((ni, no), []).append(os.path.basename(f).replace('.tl', ''))
    data_shapes = set()
    for batch in ['batch_v2_full', 'batch_v2_io', 'batch_v2_rest']:
        for f in glob.glob(f'data/{batch}/circuit_static*.parquet'):
            s = pd.read_parquet(f, columns=['input_pins_json', 'output_pins_json'])
            for _, r in s.iterrows():
                data_shapes.add((len(lj(r['input_pins_json']) or []),
                                 len(lj(r['output_pins_json']) or [])))
    print(f"  Rust 形状数: {len(rust)}")
    for (ni, no), names in sorted(rust.items(), key=lambda x: (-x[0][1], -x[0][0])):
        cov = (ni, no) in data_shapes
        print(f"  {ni}入{no}出  {len(names)}个({names[0]})  {'OK' if cov else 'MISS'}")

    # 组大小过滤视图（训练默认 MIN_GROUP_SIZE=10）
    print("\n=== 组大小过滤（训练默认剔除 <10 变体组）===")
    dyn_all = pd.concat([stats[b][1] for b in batches], ignore_index=True)
    dyn_all['circuit_id'] = dyn_all['circuit_id'].astype(str)
    gsize = dyn_all.groupby('expr')['circuit_id'].nunique()
    for N in [10, 4, 2]:
        keep = gsize[gsize >= N].index.astype(str)
        f = dyn_all[dyn_all['expr'].astype(str).isin(keep)]
        print(f"  >= {N} 变体: 样本={len(f)} 电路={f['circuit_id'].nunique()} expr={len(keep)} "
              f"(剔 {len(gsize)-len(keep)} 组)")


if __name__ == '__main__':
    main()
