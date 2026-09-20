"""V3 数据可加载性 + 宏表覆盖 + 与 V2 的 schema 差异（2026-09-20）。

为什么写：V3 交付（`data/v3_delivery/`，37 文件，远端 d2533a0）与既有 loader 的假设不一致——

  (1) **形态**：单 `circuit_static.parquet` + 31 个 `timing_arcs_partNN.parquet`（混合）。
      旧 loader 要求两侧「同时单文件」或「同时分片」，于是 `DATA_BATCHES=v3_delivery`
      会打印 `V2 data not found` 然后**一个文件都不加载**。
      `DATA_SPEC_V2.md`「parquet 可拆多文件但同属一集」⇒ 混合形态合法，是 loader 的组合假设漏了。
  (2) **宏表**：V3 自带 `data/v3_delivery/sc_expansion.json`（1,049 宏），而训练读的是
      **硬编码共享路径** `data/sc_expansion.json`（`graph_builder.py:59-63`）。
      规格 P15 要求宏表覆盖训练数据里出现的**全部** SC_ cell 名。

本脚本只做「读」与「核对」，**不改任何文件、不跑训练**。
用 pyarrow 的 footer 元数据取行数/schema/空值计数，避免把 2.9GB 读进内存。

⚠ 关键：本脚本调用的是 `src.train_sweep.resolve_v2_batch_files` —— **真代码**，
  不是在脚本里另抄一份发现逻辑。另抄一份等于只测了副本。

只读 data/ 与本仓库代码。
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
V3 = os.path.join(ROOT, 'data', 'v3_delivery')

import pyarrow.parquet as pq            # noqa: E402


def rule(t):
    print()
    print('=' * 100)
    print(t)
    print('=' * 100)


def nrows_schema(path):
    """只读 footer：行数 + 列名列表。不读数据页。"""
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, list(pf.schema_arrow.names), pf


def null_counts(pf, names):
    """从列统计拿空值计数；统计缺失记 None（**不是 0** —— 未知≠无空值）。"""
    out = {}
    md = pf.metadata
    for j, nm in enumerate(names):
        tot, ok = 0, True
        for rg in range(md.num_row_groups):
            try:
                st = md.row_group(rg).column(j).statistics
            except Exception:
                ok = False
                break
            if st is None or st.null_count is None:
                ok = False
                break
            tot += st.null_count
        out[nm] = tot if ok else None
    return out


# ---------------------------------------------------------------- 1. 真 loader
rule('1. 用**真代码** resolve_v2_batch_files 解析（这是本次补丁的验收）')
from src.train_sweep import resolve_v2_batch_files            # noqa: E402

print('--- V3（混合形态：单 static + 31 分片 arcs）---')
s3, d3 = resolve_v2_batch_files(ROOT, 'v3_delivery')
print(f'    解析出: static {len(s3)} 个 / arcs {len(d3)} 个')
if s3:
    print(f'    static[0] = {os.path.relpath(s3[0], ROOT)}')
if d3:
    print(f'    arcs[0]   = {os.path.relpath(d3[0], ROOT)}')
    print(f'    arcs[-1]  = {os.path.relpath(d3[-1], ROOT)}')

print()
print('--- V2 默认三批（回归：补丁不能改变既有行为）---')
s2, d2 = resolve_v2_batch_files(ROOT, 'batch_v2_full,batch_v2_rest,batch_v2_m4')
print(f'    解析出: static {len(s2)} 个 / arcs {len(d2)} 个')

print()
print('--- 负例：不存在的批次仍须跳过而不是崩溃 ---')
sb, db = resolve_v2_batch_files(ROOT, 'no_such_batch')
print(f"    解析出: static {len(sb)} / arcs {len(db)}  （应为 0/0 且打印 'skipping'）")

# ---------------------------------------------------------------- 2. 分片清单
rule('2. 分片行数/字节 vs PARTS.txt vs metadata.json（完整性，不读数据页）')
declared = {}
with open(os.path.join(V3, 'PARTS.txt'), encoding='utf-8') as fh:
    head = fh.readline().strip()
    print(f'    PARTS.txt 头: {head}')
    for ln in fh:
        p = ln.rstrip('\n').split('\t')
        if len(p) >= 3:
            declared[p[0]] = (int(p[1]), int(p[2]))
print(f'    PARTS.txt 声明 {len(declared)} 片')

real_rows = real_bytes = 0
mismatch = []
for name in sorted(declared):
    fp = os.path.join(V3, name)
    if not os.path.exists(fp):
        mismatch.append(f'{name}: 文件不存在')
        continue
    n, _, _ = nrows_schema(fp)
    sz = os.path.getsize(fp)
    real_rows += n
    real_bytes += sz
    dn, db = declared[name]
    if n != dn or sz != db:
        mismatch.append(f'{name}: 实际 {n}行/{sz}B vs 声明 {dn}行/{db}B')
print(f'    实测合计: {real_rows} 行 / {real_bytes} 字节')
print(f'    声明合计: {sum(v[0] for v in declared.values())} 行 / '
      f'{sum(v[1] for v in declared.values())} 字节')
print(f'    逐片不一致: {len(mismatch)} 处' + ('  ⚠ ' + '; '.join(mismatch[:5]) if mismatch else '  ✅'))

meta = json.load(open(os.path.join(V3, 'metadata.json'), encoding='utf-8'))
print(f'    metadata.json 顶层键: {sorted(meta.keys())}')
for k in ('rows', 'circuits', 'exprs', 'n_circuits', 'n_exprs'):
    if k in meta:
        print(f'      {k} = {meta[k]}')
if 'split_rows' in meta:
    print(f"      split_rows = {meta['split_rows']}  合计 "
          f"{sum(meta['split_rows'].values())}")
print(f"    ⇒ 分片行数 == metadata.rows ? "
      f"{real_rows == meta.get('rows')}")

# ---------------------------------------------------------------- 3. schema
rule('3. schema：V3 三侧 vs V2 三批（训练读的列必须都在）')
v3_static = os.path.join(V3, 'circuit_static.parquet')
n_s, s_cols, pf_s = nrows_schema(v3_static)
print(f'    V3 circuit_static: {n_s} 行 / {len(s_cols)} 列')
n_d, d_cols, pf_d = nrows_schema(d3[0])
print(f'    V3 timing_arcs 单片: {n_d} 行 / {len(d_cols)} 列')

# 分片 schema 必须完全一致，否则 concat 会错列
diff_parts = []
for p in d3:
    _, cols, _ = nrows_schema(p)
    if cols != d_cols:
        diff_parts.append(os.path.basename(p))
print(f'    31 片 schema 不一致的: {len(diff_parts)} 处' +
      (f'  ⚠ {diff_parts[:5]}' if diff_parts else '  ✅'))

v2_s_cols = v2_d_cols = None
for b in ('batch_v2_full',):
    bp = os.path.join(ROOT, 'data', b)
    sp = os.path.join(bp, 'circuit_static.parquet')
    if os.path.exists(sp):
        _, v2_s_cols, _ = nrows_schema(sp)
    dpl = sorted(glob.glob(os.path.join(bp, 'timing_arcs*.parquet')))
    if dpl:
        _, v2_d_cols, _ = nrows_schema(dpl[0])
        print(f'    V2 对照取自 {b}: static {len(v2_s_cols or [])} 列 / '
              f'arcs({"单片" if "part" not in dpl[0] else "分片[0]"}) {len(v2_d_cols)} 列')

if v2_s_cols and v2_d_cols:
    print()
    print(f'    static 列: V3 独有 = {sorted(set(s_cols) - set(v2_s_cols)) or "无"}')
    print(f'    static 列: V2 独有 = {sorted(set(v2_s_cols) - set(s_cols)) or "无"}  ← 缺列会让特征静默退化')
    print(f'    arcs   列: V3 独有 = {sorted(set(d_cols) - set(v2_d_cols)) or "无"}')
    print(f'    arcs   列: V2 独有 = {sorted(set(v2_d_cols) - set(d_cols)) or "无"}  ← 缺列会让特征静默退化')

need_d = ['circuit_id', 'expr', 'switching_pin', 'direction', 'DELAY', 'vector',
          'slew_s', 'output_load_f']
print(f'    训练必需(arcs) 缺失: {[c for c in need_d if c not in d_cols] or "无"}')

# ---------------------------------------------------------------- 4. 空值
rule('4. 空值计数（取自列统计；None = 该列无统计，属**未验证**不是"无空值"）')
nc = null_counts(pf_d, d_cols)
for c in need_d + ['transistor_wave_json', 'gate_states', 'supply_noise',
                   'pin_slew', 'pin_load', 'ids_avg', 'ids_charge']:
    if c in nc:
        v = nc[c]
        print(f'    {c:<24s} 空值 = {"未验证(无列统计)" if v is None else v}')

# ---------------------------------------------------------------- 5. 宏表
rule('5. sc_expansion.json 覆盖：V3 表 vs 旧共享表（规格 P15 要求 100% 可展开）')
from src.graph_builder import _asap7_cell_feat                # noqa: E402

ct = pq.read_table(v3_static, columns=['cell_types_json']).to_pylist()
names = []
for r in ct:
    try:
        v = json.loads(r['cell_types_json'])
    except Exception:
        continue
    if isinstance(v, list):
        names.extend(str(x) for x in v)
    elif isinstance(v, dict):
        names.extend(str(k) for k in v)
uniq = sorted(set(names))
sc_cells = [n for n in uniq if n.startswith('SC_')]
print(f'    V3 静态表 cell 名: 唯一 {len(uniq)} 个，其中 SC_ 前缀 {len(sc_cells)} 个')


def coverage(path, label, cells):
    try:
        tbl = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        print(f'    [{label}] 读不了: {e}')
        return
    present = exp_ok = 0
    null_or_empty = 0
    bad_cell = []
    for n in cells:
        e = tbl.get(n)
        if e is None:
            continue
        present += 1
        sub = e.get('subcircuit') if isinstance(e, dict) else None
        if not sub:
            null_or_empty += 1
            continue
        feats = [f for f in (_asap7_cell_feat(x.get('cell', '')) for x in sub) if f is not None]
        if feats:
            exp_ok += 1
        else:
            bad_cell.append(n)
    tot = len(cells)
    print(f'    [{label}] 表内条目 {len(tbl)}')
    print(f'        存在率   {present}/{tot} = {present / tot:.1%}')
    print(f'        subcircuit 为 null/空 {null_or_empty}')
    print(f'        可展开率 {exp_ok}/{tot} = {exp_ok / tot:.1%}   ← 这个才决定特征是否退化')
    if bad_cell:
        print(f'        展开不出 ASAP7 单元的示例: {bad_cell[:3]}')


# ⚠ 双向都要测：共享表是**全局**的，覆盖它会影响**未来的 V2 运行**。
#   只测「V3 表对 V3 名域」会漏掉「V3 表对 V2 名域」这个副作用。
def cell_names_of(path, label):
    if not os.path.exists(path):
        print(f'    [{label}] 无文件，跳过')
        return []
    tbl = pq.read_table(path, columns=['cell_types_json']).to_pylist()
    out = []
    for r in tbl:
        try:
            v = json.loads(r['cell_types_json'])
        except Exception:
            continue
        if isinstance(v, list):
            out.extend(str(x) for x in v)
        elif isinstance(v, dict):
            out.extend(str(k) for k in v)
    return sorted(set(n for n in out if n.startswith('SC_')))


V3_TBL = os.path.join(V3, 'sc_expansion.json')
SHARED = os.path.join(ROOT, 'data', 'sc_expansion.json')
v2_cells = cell_names_of(os.path.join(ROOT, 'data', 'batch_v2_full', 'circuit_static.parquet'),
                         'V2 full')
print(f'    V2 名域 SC_ 唯一 {len(v2_cells)} 个')

print()
print('    --- A) 对 V3 名域（决定 V3 训练是否退化）---')
coverage(V3_TBL, 'V3 自带表', sc_cells)
coverage(SHARED, '当前共享表', sc_cells)
print()
print('    --- B) 对 V2 名域（决定**覆盖共享路径后 V2 会不会被带崩**）---')
if v2_cells:
    coverage(V3_TBL, 'V3 自带表', v2_cells)
    coverage(SHARED, '当前共享表', v2_cells)

# 额外表（用 SC_EXPANSION_EXTRA=a.json,b.json 给）——用于验收合并结果
# `scripts/merge_sc_expansion.py --out` 的产物：期望 A 与 B 两块都变 100%。
for p in [x for x in os.environ.get('SC_EXPANSION_EXTRA', '').split(',') if x.strip()]:
    if not os.path.exists(p):
        print(f'    [{p}] 不存在，跳过')
        continue
    print()
    print(f'    --- 额外表 {p} ---')
    coverage(p, os.path.basename(p), sc_cells)
    if v2_cells:
        coverage(p, os.path.basename(p), v2_cells)

# ---------------------------------------------------------------- 6. 管数口径
rule('6. transistor_count 口径：V3（每实例展开）vs V2（宏去重）—— 交付方 round 10 声明换过口径')
for label, path in (('V3', v3_static),
                    ('V2 batch_v2_full', os.path.join(ROOT, 'data', 'batch_v2_full', 'circuit_static.parquet'))):
    if not os.path.exists(path):
        print(f'    [{label}] 无文件，跳过')
        continue
    col = 'transistor_count'
    if col not in pq.ParquetFile(path).schema_arrow.names:
        print(f'    [{label}] 无 {col} 列')
        continue
    vals = sorted(int(x) for x in pq.read_table(path, columns=[col]).column(0).to_pylist()
                  if x is not None)
    n = len(vals)
    print(f'    [{label}] n={n}  p10={vals[n // 10]}  p50={vals[n // 2]}  '
          f'p90={vals[9 * n // 10]}  max={vals[-1]}')

# ---------------------------------------------------------------- 7. 组大小
rule('7. 分组（expr）规模 —— 训练默认 MIN_GROUP_SIZE 过滤前的分布')
try:
    import pandas as pd
    st = pq.read_table(v3_static, columns=['circuit_id', 'expr']).to_pandas()
    st['circuit_id'] = st['circuit_id'].astype(str)
    g = st.groupby('expr')['circuit_id'].nunique()
    print(f'    V3: 电路 {len(st)} / expr 组 {len(g)}')
    print(f'        gsize 分布 = {dict(sorted(Counter(g.tolist()).items()))}')
    print(f'        gsize 中位 {g.median():.0f}  最小 {g.min()}  最大 {g.max()}')
    print(f'        < {10} 的组: {int((g < 10).sum())} / {len(g)}')
except Exception as e:
    print(f'    跳过: {e}')

print()
print('完成（本脚本未修改任何文件）')
