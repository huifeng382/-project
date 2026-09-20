"""V3 交付 vs `docs/DATA_SPEC_V2.md` §四「形状-规模-深度锚定」逐条验收（2026-09-20）。

为什么写：用户要求「检查一下这个版本的数据是否满足之前的数据生成要求」。
要求有两处：DIFF §12.4 缺陷清单 + `DATA_SPEC_V2.md` §四正文（后者是正式规格）。
**规格以 §四为准**（§12.4 是分析依据，且其 1/2 条 2026-09-03 已降为「统计参考」）。

关键：**深度必须自己算**。metadata.json 的 `shape_coverage` 只给了
`{n_circuits, trans_p50, trans_p90, trans_max, rows, rust_band, in_band_share,
p50_ge_rust_med}` —— **没有 `depth_p50/depth_p90`**，而规格验收条款 3
（★深主档 ≥30% ≥9 且 p90≥10）和 §四量化验收②（「深度分位数」）都要求它。
⇒ 本脚本从 `gate_level_netlist` 按规格定的口径实测：
   **深度 = X_ 宏级最长链**（§四「深度口径（2026-09-03 统一）」，SC_JOIN_* 宏算 1 层，
   不是 .tl 模板的 X 级联）。

同时**独立复核** metadata.json 自己的说法（trans 分位、逐形状 n_circuits、Tier 占比、
弱驱动占比），不采信自评——这是本项目一贯做法。

读法：static 全列（7.6MB）+ arcs **只读 circuit_id 一列**（为算每电路行数），
不把 2.9GB 读进内存。只读，不改任何文件，不跑训练。
"""
import json
import os
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V3 = os.path.join(ROOT, 'data', 'v3_delivery')

import numpy as np                      # noqa: E402
import pyarrow.parquet as pq            # noqa: E402

# ── Rust 候选实测锚定表（`DATA_SPEC_V2.md` §四「锚定清单（18 种 Rust 实测形状）」
#    + 「强制规模档」+「强制深度档」）。trans = (Rust med, Rust max)，depth = (med, max)。
#    quota = 表内「V3 最低电路数」。
RUST = {
    (1, 1):  dict(name='INV1',            trans=(4, 8),     depth=(2, 2),   cat='普通',                 quota=300, star_scale=False, star_deep=None),
    (2, 1):  dict(name='OR2/AND2/XOR2…',  trans=(8, 51),    depth=(None, 5), cat='普通',                quota=400, star_scale=False, star_deep=None),
    (2, 2):  dict(name='HA',              trans=(26, 36),   depth=(4, 5),   cat='普通 多输出',           quota=300, star_scale=False, star_deep=None),
    (2, 3):  dict(name='COMP1',           trans=(29, 32),   depth=(3, 3),   cat='普通 多输出',           quota=250, star_scale=False, star_deep=None),
    (3, 1):  dict(name='OAI21/AOI21/MUX2…', trans=(11, 39), depth=(None, 6), cat='普通',                quota=400, star_scale=False, star_deep=None),
    (3, 2):  dict(name='FA/CSA_3_2',      trans=(49, 50),   depth=(6, 6),   cat='普通 多输出',           quota=350, star_scale=False, star_deep=None),
    (4, 1):  dict(name='AOI22/…/PARITY4', trans=(12, 52),   depth=(3, 8),   cat='◆深次档',               quota=400, star_scale=False, star_deep='sub'),
    (4, 3):  dict(name='ENC4',            trans=(56, 69),   depth=(None, 6), cat='普通 多输出(宽松下限)', quota=400, star_scale=False, star_deep=None),
    (5, 1):  dict(name='AOI221/…/DEPTH_MIX', trans=(14, 117), depth=(2, 6), cat='普通(spread大变体)',    quota=800, star_scale=False, star_deep=None),
    (5, 2):  dict(name='ALU_SLICE_SMALL', trans=(140, 208), depth=(6, 11),  cat='★规模+◆深次档',         quota=900, star_scale=True,  star_deep='sub'),
    (5, 5):  dict(name='SHIFTER4',        trans=(48, 60),   depth=(3, 3),   cat='普通 多输出',           quota=300, star_scale=False, star_deep=None),
    (7, 4):  dict(name='ALU2',            trans=(235, 262), depth=(9, 10),  cat='★规模+★深主档',         quota=550, star_scale=True,  star_deep='main'),
    (8, 1):  dict(name='AND8/OR8',        trans=(22, 67),   depth=(2, 4),   cat='普通',                  quota=300, star_scale=False, star_deep=None),
    (8, 3):  dict(name='COMP4',           trans=(170, 233), depth=(10, 12), cat='★规模+★深主档',         quota=650, star_scale=True,  star_deep='main'),
    (8, 4):  dict(name='ENC8',            trans=(189, 233), depth=(4, 4),   cat='★规模(大而浅,勿要求深)', quota=500, star_scale=True,  star_deep=None),
    (9, 1):  dict(name='OVF',             trans=(104, 146), depth=(13, 15), cat='★深主档(中规模)',       quota=450, star_scale=False, star_deep='main'),
    (9, 6):  dict(name='ADD4_OVF',        trans=(247, 316), depth=(12, 16), cat='★规模+★深主档',         quota=350, star_scale=True,  star_deep='main'),
    (16, 1): dict(name='AND16',           trans=(35, 40),   depth=(2, 2),   cat='普通',                  quota=300, star_scale=False, star_deep=None),
}
# 非 ★/◆ 的多输出锚形状 —— 规格「宽松下限：trans p50 ≥ Rust med × 0.7」
LOOSE_MULTI = [(2, 2), (2, 3), (3, 2), (4, 3), (5, 5)]


def rule(t):
    print()
    print('=' * 108)
    print(t)
    print('=' * 108)


def pctl(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')


# ═════════════════════════════════════════════════════ 0. 读 static
rule('0. 读入 static（全列）+ arcs 只读 circuit_id（算每电路行数）')
sp = os.path.join(V3, 'circuit_static.parquet')
st = pq.read_table(sp).to_pylist()
print(f'    static: {len(st)} 行')
ids = [str(r['circuit_id']) for r in st]
dup = len(ids) - len(set(ids))
print(f'    circuit_id 唯一性: {len(set(ids))} 个唯一 / 重复 {dup} 处'
      + ('  ✅' if dup == 0 else '  ⚠'))

import glob                              # noqa: E402
parts = sorted(glob.glob(os.path.join(V3, 'timing_arcs_part*.parquet')))
rowcnt = Counter()
dsum, dcnt = defaultdict(float), Counter()      # 逐电路 DELAY 均值（供 §8 组内差用）
for k, p in enumerate(parts):
    t = pq.read_table(p, columns=['circuit_id', 'DELAY']).to_pylist()
    for r in t:
        c = str(r['circuit_id'])
        rowcnt[c] += 1
        if r['DELAY'] is not None:
            dsum[c] += float(r['DELAY'])
            dcnt[c] += 1
    if (k + 1) % 10 == 0:
        print(f'      已读 {k + 1}/{len(parts)} 片…')
tot_rows = sum(rowcnt.values())
print(f'    arcs: {tot_rows} 行（{len(parts)} 片）')
print(f'    每电路行数: 中位 {np.median(list(rowcnt.values())):.0f}  '
      f'min {min(rowcnt.values())}  max {max(rowcnt.values())}')
missing = [c for c in ids if c not in rowcnt]
print(f'    static 里有、arcs 里无行的电路: {len(missing)} 个'
      + ('  ✅' if not missing else f'  ⚠ 例 {missing[:3]}'))


# ═════════════════════════════════════════════════════════ 1. 深度（X_ 宏级最长链）
def parse_dut(nl):
    """只取 `.SUBCKT DUT … .ENDS` 块内的 X_ 行（规格：跳过嵌套 .SUBCKT 与 M_ 行）。"""
    gates = []
    indut = False
    for raw in nl.replace('\r', '\n').split('\n'):
        s = raw.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith('.SUBCKT'):
            t = s.split()
            indut = len(t) > 1 and t[1].upper() == 'DUT'
            continue
        if u.startswith('.ENDS'):
            if indut:
                break
            continue
        if not indut or not u.startswith('X'):
            continue
        t = s.split()
        if len(t) < 3:
            continue
        gates.append((t[-2], t[1:-2], t[-1]))      # (out_net, in_nets, cell)
    return gates


def dut_depths(gates, out_names):
    """返回 (逐门 level, 电路深度, 环计数, 生产者表)。深度 = 从输入到输出的最大 X_ 门链长。"""
    prod = defaultdict(list)
    for i, (o, _, _) in enumerate(gates):
        prod[o].append(i)
    memo, cyc = {}, [0]

    def lvl(i, stack):
        if i in memo:
            return memo[i]
        if i in stack:                            # 组合环（不该有）——不静默吞
            cyc[0] += 1
            return 0
        stack.add(i)
        ins = gates[i][1]
        m = 0
        for n in ins:
            for p in prod.get(n, ()):
                v = lvl(p, stack)
                if v > m:
                    m = v
        stack.discard(i)
        memo[i] = m + 1
        return m + 1

    allv = [lvl(i, set()) for i in range(len(gates))]
    to_out = [allv[i] for i, (o, _, _) in enumerate(gates) if o in out_names]
    d = max(to_out) if to_out else (max(allv) if allv else 0)
    return allv, d, cyc[0], prod


rule('1. 深度实测（口径 = 规格 §四：X_ 宏级最长链，SC_JOIN_* 宏算 1 层）')
per = []
cyc_tot = bad_parse = 0
for r in st:
    ins = json.loads(r['input_pins_json'])
    outs = json.loads(r['output_pins_json'])
    g = parse_dut(r['gate_level_netlist'])
    lv, d, c, prod = dut_depths(g, set(outs))
    cyc_tot += c
    if not g:
        bad_parse += 1
    per.append(dict(cid=str(r['circuit_id']), expr=str(r['expr']),
                    nin=len(ins), nout=len(outs),
                    shape=(len(ins), len(outs)),
                    shape_col=str(r.get('shape')),
                    trans=int(r['transistor_count']),
                    depth=d, nx=len(g), tier=str(r.get('tier')),
                    gates=g, prod=prod, outs=set(outs),
                    par=r.get('parasitic_caps_json'),
                    rows=rowcnt.get(str(r['circuit_id']), 0)))
dv = [x['depth'] for x in per]
print(f'    电路 {len(per)} 个；解析不出任何 X_ 门的: {bad_parse} 个')
print(f'    组合环: {cyc_tot} 处' + ('  ✅' if cyc_tot == 0 else '  ⚠ 拓扑异常'))
print(f'    深度: p10 {pctl(dv, 10):.0f}  p50 {pctl(dv, 50):.0f}  p90 {pctl(dv, 90):.0f}  '
      f'max {max(dv)}')
print(f'    depth >6 占比: {np.mean([d > 6 for d in dv]):.1%}   '
      f'（§四第 2 条「≥10% 深 >6」= 统计参考，2026-09-03 二版降级）')
print(f'    depth >18（Rust 域外线）: {np.mean([d > 18 for d in dv]):.2%}')
nxf = [x['nx'] for x in per]
print(f'    X_ 门数（宏实例数）: med {np.median(nxf):.0f}  p90 {pctl(nxf, 90):.0f}  max {max(nxf)}'
      f'   ← 规格 §四第 3 条要求落 Rust 量级 med 32 / p90 57 / max 73')
# shape 列 vs 现算
mis = sum(1 for x in per if x['shape_col'] not in ('', 'None') and
          x['shape_col'] not in (f'{x["nin"]}x{x["nout"]}', f'{x["nin"]},{x["nout"]}'))
print(f'    交付 shape 列与现算 (N_in,N_out) 不一致: {mis} 处'
      + ('  ✅' if mis == 0 else '  ⚠') + f'（shape 列样例 {per[0]["shape_col"]!r}）')


# ═════════════════════════════════════════════════════════ 2. 逐形状验收
rule('2. 逐形状验收（规格 §四「锚定清单」+「量化验收」1/2/3/4 条）')
by = defaultdict(list)
for x in per:
    by[x['shape']].append(x)
meta = json.load(open(os.path.join(V3, 'metadata.json'), encoding='utf-8'))
msc = meta.get('shape_coverage', {})

hdr = (f'{"形状":>7} {"Rust代表":<12} {"n_电路":>6} {"配额":>5} {"行数":>8} '
       f'{"trans p50":>9} {"p90":>5} {"max":>5} {"depth p50":>9} {"p90":>4} {"max":>4} '
       f'{"≥7":>6} {"≥9":>6}  判定')
print(hdr)
print('-' * len(hdr))
fails = []
for sh in sorted(RUST, key=lambda t: (t[1], t[0])):
    ref = RUST[sh]
    g = by.get(sh, [])
    if not g:
        print(f'{f"{sh[0]}x{sh[1]}":>7} {ref["name"][:12]:<12} {"—":>6} {ref["quota"]:>5} '
              f'{"—":>8} {"—":>9} {"—":>5} {"—":>5} {"—":>9} {"—":>4} {"—":>4} '
              f'{"—":>6} {"—":>6}  ❌ 该形状一个电路都没有')
        fails.append(f'{sh[0]}x{sh[1]}: 形状缺失')
        continue
    tr = [x['trans'] for x in g]
    dp = [x['depth'] for x in g]
    rw = sum(x['rows'] for x in g)
    n = len(g)
    ge7 = float(np.mean([d >= 7 for d in dp]))
    ge9 = float(np.mean([d >= 9 for d in dp]))
    v = []
    # 量化验收 1：n_circuits ≥ 配额
    if n < ref['quota']:
        v.append(f'n<{ref["quota"]}❌')
    # 量化验收 2：★规模五形状
    if ref['star_scale']:
        if pctl(tr, 50) < ref['trans'][0]:
            v.append('p50<med❌')
        if max(tr) < ref['trans'][1] * 0.85:
            v.append('max<0.85❌')
    # 量化验收 4：非★/◆ 多输出宽松下限 p50 ≥ med×0.7
    if sh in LOOSE_MULTI and pctl(tr, 50) < ref['trans'][0] * 0.7:
        v.append('宽松下限❌')
    # 量化验收 3：深度档
    if ref['star_deep'] == 'main':
        if ge9 < 0.30:
            v.append(f'≥9仅{ge9:.0%}<30%❌')
        if pctl(dp, 90) < 10:
            v.append(f'depth_p90={pctl(dp, 90):.0f}<10❌')
    if ref['star_deep'] == 'sub' and ge7 < 0.20:
        v.append(f'≥7仅{ge7:.0%}<20%❌')
    # 配额表内的附注要求
    if sh == (4, 1) and sum(1 for d in dp if d >= 7) < 80:
        v.append(f'深≥7仅{sum(1 for d in dp if d >= 7)}<80❌')
    if sh == (9, 1) and sum(1 for d in dp if d >= 10) < 150:
        v.append(f'深≥10仅{sum(1 for d in dp if d >= 10)}<150❌')
    verdict = '✅' if not v else '❌ ' + ' '.join(v)
    if v:
        fails.append(f'{sh[0]}x{sh[1]}: ' + ' '.join(v))
    print(f'{f"{sh[0]}x{sh[1]}":>7} {ref["name"][:12]:<12} {n:>6} {ref["quota"]:>5} {rw:>8} '
          f'{pctl(tr, 50):>9.0f} {pctl(tr, 90):>5.0f} {max(tr):>5} {pctl(dp, 50):>9.0f} '
          f'{pctl(dp, 90):>4.0f} {max(dp):>4} {ge7:>6.0%} {ge9:>6.0%}  {verdict}')

extra = [sh for sh in by if sh not in RUST]
print()
print(f'    表外形状（18 种之外）: {len(extra)} 种 —— '
      f'{sorted(((f"{a}x{b}", len(by[(a, b)])) for a, b in extra), key=lambda t: -t[1])}')
oob = [sh for sh in by if sh[0] > 16 or sh[1] > 6]
print(f'    Rust 可达域外（>16 入 或 >6 出）: {len(oob)} 种' + ('  ✅' if not oob else f'  ⚠ {oob}'))

print()
print(f'    ── 判定：18 锚形状中 {18 - len([f for f in fails if "形状缺失" in f])}/18 存在，'
      f'不合格 {len(fails)} 项 ──')
for f in fails:
    print(f'      ❌ {f}')


# ═════════════════════════════════════════════════════════ 3. 规模档 / 深度档全局
rule('3. 强制规模档两档 + 深度档 + Tier A/B（规格 §四「三档预算」+「量化验收」5）')
sc5 = None
star5 = [(5, 2), (7, 4), (8, 3), (8, 4), (9, 6)]
band_189_233 = band_233_316 = 0
for sh in star5:
    for x in by.get(sh, []):
        if 189 <= x['trans'] <= 233:
            band_189_233 += 1
        if 233 <= x['trans'] <= 316:
            band_233_316 += 1
print(f'    ★规模五形状内 189-233 档: {band_189_233} 电路   233-316 档: {band_233_316} 电路'
      f'   （规格要求「两档都要有」）')

tier_r = Counter()
tier_c = Counter()
for x in per:
    tier_r[x['tier']] += x['rows']
    tier_c[x['tier']] += 1
print(f'    tier 列（交付自带）: 按电路 {dict(tier_c)}；按行 {dict(tier_r)}')
if tier_r:
    tA = tier_r.get('A', tier_r.get('tierA', 0)) / max(1, sum(tier_r.values()))
    print(f'    Tier A 行占比实测 {tA:.1%}（规格目标 ~37%）')
    print(f'    metadata 自评 tierA_row_share = {meta.get("tier", {}).get("tierA_row_share")}')

inA = [x for x in per if x['tier'].upper().endswith('A')]
inB = [x for x in per if x['tier'].upper().endswith('B')]
print()
print(f'    Tier B 分布判据（规格：trans p50 ≥120、140-190 带 ≥20% 行、深 6-8 带 ≥15% 行）:')
if inB:
    btr = [x['trans'] for x in inB]
    bdp = [x['depth'] for x in inB]
    b_rows = sum(x['rows'] for x in inB)
    w = np.array([x['rows'] for x in inB], dtype=float)
    share_140_190 = (w[(np.array(btr) >= 140) & (np.array(btr) <= 190)].sum() / b_rows) if b_rows else 0
    share_d68 = (w[(np.array(bdp) >= 6) & (np.array(bdp) <= 8)].sum() / b_rows) if b_rows else 0
    print(f'      Tier B 电路 {len(inB)} / {b_rows} 行')
    print(f'      trans p50（按电路）{pctl(btr, 50):.1f}  ≥120 ? '
          f'{"✅" if pctl(btr, 50) >= 120 else "❌"}')
    print(f'      140-190 带行占比 {share_140_190:.1%}  ≥20% ? '
          f'{"✅" if share_140_190 >= 0.20 else "❌"}')
    print(f'      深 6-8 带行占比 {share_d68:.1%}  ≥15% ? '
          f'{"✅" if share_d68 >= 0.15 else "❌"}')
else:
    print('      ⚠ tier 列里没有 B')

print()
print('    Tier A / B 各自画像（规格说 Tier B 是「衔接 Tier A 的过渡带」，须验证）：')
for lab, g in (('Tier A', inA), ('Tier B', inB)):
    if not g:
        continue
    tr = [x['trans'] for x in g]
    dp = [x['depth'] for x in g]
    tw = np.array(tr, dtype=float)
    print(f'      {lab}: {len(g)} 电路 / {sum(x["rows"] for x in g)} 行')
    print(f'        trans 按电路 p50 {pctl(tr, 50):.0f} / p90 {pctl(tr, 90):.0f} / max {max(tr)}'
          f'   按行 p50 {np.percentile(tw, 50):.0f}')
    print(f'        depth 按电路 p50 {pctl(dp, 50):.0f} / p90 {pctl(dp, 90):.0f} / max {max(dp)}')
    sh = Counter(x['shape'] for x in g)
    top = '  '.join(f'{a}x{b}:{c}' for (a, b), c in sh.most_common(6))
    print(f'        形状构成 top6: {top}')

print()
print('    联合分布（行数口径；看 141-190 管 / 深 6-8 这两条「过渡带」是否真空）：')
TB = [(0, 100), (101, 140), (141, 190), (191, 233), (234, 316), (317, 400)]
DB = [(0, 5), (6, 8), (9, 12), (13, 18)]
print(f'      {"trans\\depth":>12} ' + ' '.join(f'{f"{a}-{b}":>9}' for a, b in DB) + f'{"合计":>9}')
for lo, hi in TB:
    cells = [sum(x['rows'] for x in per
                 if lo <= x['trans'] <= hi and dlo <= x['depth'] <= dhi) for dlo, dhi in DB]
    print(f'      {f"{lo}-{hi}":>12} ' + ' '.join(f'{c:>9}' for c in cells) + f'{sum(cells):>9}')
print(f'      {"合计":>12} ' + ' '.join(
    f'{sum(x["rows"] for x in per if a <= x["depth"] <= b):>9}' for a, b in DB)
    + f'{sum(x["rows"] for x in per):>9}')
print('      （规格 Tier B 判据要求：141-190 管带 ≥20% 行、深 6-8 带 ≥15% 行）')

# 「浅 / 小」区间还有多少 —— V3 是唯一训练数据后，这一格决定旧数据那批主流模板还在不在
print()
print('    「浅/小」区间的余量（旧数据 = 深 ≤5 / trans ≤104 为主；V3 现在是唯一训练数据）：')
for nm, f in (('深 ≤5', lambda x: x['depth'] <= 5),
              ('深 ≤6', lambda x: x['depth'] <= 6),
              ('trans ≤100', lambda x: x['trans'] <= 100),
              ('trans ≤104', lambda x: x['trans'] <= 104),
              ('深 ≤5 且 trans ≤104', lambda x: x['depth'] <= 5 and x['trans'] <= 104),
              ('深 ≥9', lambda x: x['depth'] >= 9),
              ('trans ≥180', lambda x: x['trans'] >= 180)):
    g = [x for x in per if f(x)]
    print(f'      {nm:<20s} 电路 {len(g):>6} ({len(g) / len(per):>5.1%})   '
          f'行 {sum(x["rows"] for x in g):>7} ({sum(x["rows"] for x in g) / tot_rows:>5.1%})')

print()
rwo = meta.get('row_weighted', {})
print()
print(f'    metadata row_weighted（行口径，自评）: {rwo}')
ar = np.array([x['rows'] for x in per], dtype=float)
atr = np.array([x['trans'] for x in per], dtype=float)
ok = ar > 0
print(f'    实测行口径 trans p50 {np.percentile(atr[ok], 50):.0f}  '
      f'180-400 管行占比 {ar[ok][(atr[ok] >= 180) & (atr[ok] <= 400)].sum() / ar[ok].sum():.1%}'
      f'   （旧的全局「≥10% 180~400+」= 统计参考）')


# ═════════════════════════════════════════════════════════ 4. I/O 分桶
rule('4. I/O 分桶等比例（规格 §四「I/O 多样性要求：1~2 / 3~4 / 5~8 / 9~16 各约 25%」）')
BUCK = [(1, 2), (3, 4), (5, 8), (9, 16)]
print(f'    {"桶":>8} {"电路数":>7} {"电路占比":>9} {"行数":>8} {"行占比":>8}')
totc, totr = len(per), sum(x['rows'] for x in per)
for lo, hi in BUCK:
    g = [x for x in per if lo <= x['nin'] <= hi]
    r = sum(x['rows'] for x in g)
    print(f'    {f"{lo}-{hi}":>8} {len(g):>7} {len(g) / totc:>9.1%} {r:>8} {r / totr:>8.1%}')
print(f'    （规格原文：分桶约束作用于 Tier B 与整体统计，Tier A 不受分桶约束）')


# ═════════════════════════════════════════════════════════ 5. 弱驱动中间门
rule('5. 弱驱动中间门（规格 §四第 4 条：★深/大档形状内占比 ≥30%）')
wd_shape = [(5, 2), (7, 4), (8, 3), (8, 4), (9, 1), (9, 6)]
fanout_all, circ_weak_f, circ_weak_p, circ_weak_any = [], 0, 0, 0
n_wd = 0
for x in per:
    if x['shape'] not in wd_shape:
        continue
    n_wd += 1
    gates, prod, outs = x['gates'], x['prod'], x['outs']
    caps = []
    try:
        pj = json.loads(x['par']) if x['par'] else {}
    except Exception:
        pj = {}
    for v in (pj.values() if isinstance(pj, dict) else []):
        if isinstance(v, dict):
            for kv in v.values():
                try:
                    caps.append(float(kv))
                except Exception:
                    pass
    # ⚠ 扇出 = 该门输出网被**下游门**引用的次数（消费者数），不是生产者数。
    #   每个网的驱动者恒为 1（良构网表），拿 len(prod[net]) 当扇出会恒得 1、
    #   使判据近于恒真 —— 本脚本初版就踩了这个坑，故此处显式算消费者。
    use = Counter()
    for (o, ins, _) in gates:
        for n in ins:
            use[n] += 1
    wf = wp = False
    for i, (o, _, _) in enumerate(gates):
        fo = use.get(o, 0)
        fanout_all.append(fo)
        if o not in outs and fo <= 2:            # 中间门 + 扇出 ≤2
            wf = True
    med = float(np.median(caps)) if caps else 0.0
    if med > 0 and any(c >= 2 * med for c in caps):
        wp = True
    circ_weak_f += wf
    circ_weak_p += wp
    circ_weak_any += (wf or wp)
if n_wd:
    print(f'    ★深/大档形状电路 {n_wd} 个')
    print(f'      含「中间门扇出 ≤2」的电路占比  {circ_weak_f / n_wd:.1%}   '
          f'（判据之一，≥30% 即过）')
    print(f'      含「寄生 ≥ 本电路寄生中位 2×」 {circ_weak_p / n_wd:.1%}   （判据之二）')
    print(f'      二者取或（规格原判据）        {circ_weak_any / n_wd:.1%}   ≥30% ? '
          f'{"✅" if circ_weak_any / n_wd >= 0.30 else "❌"}')
    print()
    fd = [round(float(np.mean([f == k for f in fanout_all])), 3) for k in list(range(8))]
    fd.append(round(float(np.mean([f >= 8 for f in fanout_all])), 4))
    print(f'    门实例扇出分布（消费者数）[0,1,2,3,4,5,6,7,≥8] 占比 = {fd}')
    print(f'      扇出中位 {np.median(fanout_all):.0f}  mean {np.mean(fanout_all):.2f}')
    print(f'      若绝大多数门天然扇出 ≤2，该判据**近于恒真**，不构成对「弱驱动」的实际筛选'
          f'（metadata 自评 weak_drive={meta.get("weak_drive")}）')
else:
    print('    ⚠ 没有 ★深/大档形状电路')


# ═════════════════════════════════════════════════════════ 6. 样本量/组/行数上限
rule('6. 样本量 / 每组变体数 / 每电路行数上限（规格 §四表格）')
expr = Counter(x['expr'] for x in per)
gs = np.array(list(expr.values()))
print(f'    电路 {len(per)} / expr 组 {len(expr)} / 行 {tot_rows}')
print(f'    每组变体数: min {gs.min()}  p50 {np.median(gs):.0f}  max {gs.max()}   '
      f'在 10~15 内 ? {"✅" if gs.min() >= 10 and gs.max() <= 15 else "⚠"}')
bad_cap = [x for x in per if x['rows'] > 2 * x['nin'] * x['nout']]
print(f'    每电路行数 > 2×N_in×M（规格上限）: {len(bad_cap)} 个'
      + ('  ✅' if not bad_cap else f'  ⚠ 例 {[(x["cid"], x["rows"], f"{x["nin"]}x{x["nout"]}") for x in bad_cap[:3]]}'))
print(f'    隐含平均行/电路 {tot_rows / len(per):.1f}（规格估算 ≈12）')
srow = meta.get('split_rows') or meta.get('dataset', {}).get('split_rows', {})
print(f'    交付 split 列行数 {srow}（⚠ 训练用自切划分，此列无读者）')
print(f'    corner 单值: {meta.get("corner") or meta.get("dataset", {}).get("corner")}')

# ── metadata 自评 vs 实测（trans 分位）
rule('7. metadata 自评 vs 本次实测（trans 逐形状，取有差异的列出）')
nd = 0
for sh, ref in sorted(RUST.items(), key=lambda t: t[1]['name']):
    k = f'{sh[0]}x{sh[1]}'
    m = msc.get(k)
    if not m or sh not in by:
        continue
    g = by[sh]
    a50, a90, amx = pctl([x['trans'] for x in g], 50), pctl([x['trans'] for x in g], 90), max(x['trans'] for x in g)
    d = (abs(a50 - m.get('trans_p50', -1)), abs(a90 - m.get('trans_p90', -1)), abs(amx - m.get('trans_max', -1)))
    flag = '' if max(d) <= 1 else '  ⚠'
    if flag:
        nd += 1
    print(f'    {k:>6}  自评 p50/p90/max = {m.get("trans_p50")}/{m.get("trans_p90")}/{m.get("trans_max")}'
          f'   实测 {a50:.0f}/{a90:.0f}/{amx}{flag}')
for sh in [s for s in by if s not in RUST]:
    k = f'{sh[0]}x{sh[1]}'
    m = msc.get(k)
    if m:
        print(f'    {k:>6}  （表外形状）自评 n_circuits={m.get("n_circuits")} 实测 {len(by[sh])}')
print(f'    逐形状 trans 分位不一致: {nd} 处')


# ═════════════════════════════════════════════════════════ 8. 组内延迟差（缺陷 4）
rule('8. 组内变体延迟差（规格 §四第 5 条三条硬指标 + DIFF §12.4 缺陷 4）')
# 口径与 `_t_v3_group_spread.py` **逐字一致**（同一份数据两个脚本必须说同一句话）：
#   逐电路标量 = 该电路全部行 DELAY 的均值；组内升序；gap_i = (d[i+1]-d[i])/d[i]；
#   spread = (max-min)/median。
per_d = {}
for x in per:
    c = x['cid']
    if dcnt[c]:
        per_d.setdefault(x['expr'], []).append(dsum[c] / dcnt[c])
gaps, spreads, n_ge2_10, n_ge2_20, _g = [], [], 0, 0, 0
best_gap = []                      # 每组「次优相对最优」的差（组内最易区分的那一步）
for e, v in per_d.items():
    if len(v) < 2:
        continue
    _g += 1
    v.sort()
    med = v[len(v) // 2]
    if med > 0:
        spreads.append((v[-1] - v[0]) / med)
        best_gap.append((v[1] - v[0]) / v[0] if v[0] > 0 else 0.0)
    for i in range(len(v) - 1):
        if v[i] > 0:
            gaps.append((v[i + 1] - v[i]) / v[i])
    nb = v[0]
    if sum(1 for d in v if d > 1.1 * nb) >= 2:
        n_ge2_10 += 1
    if sum(1 for d in v if d > 1.2 * nb) >= 2:
        n_ge2_20 += 1
n = len(gaps)
print(f'    组 {_g} 个；逐电路标量 = 该电路全部行 DELAY 均值；相邻步长 n={n}')
print(f'    ① 「≥80% 组含 ≥2 个与组内最优差 >10% 的变体」 实测 '
      f'{n_ge2_10 / _g:.1%}  ≥80% ? {"✅" if n_ge2_10 / _g >= 0.80 else "❌"}'
      f'   （>20% 版本 {n_ge2_20 / _g:.1%}）')
print(f'        ⚠ metadata **没有报这一条**（它报的是 spread>10% 的组占比 '
      f'{meta.get("group_delay", {}).get("groups_with_gt10pct_spread")}，两者定义不同）')
print(f'    ② 相邻差四档（目标 20/35/39/6% ±10pp → 10-30 / 25-45 / 30-50 / 3-10）:')
tgt = [('<1%', 0.0, 0.01, (0.10, 0.30)), ('1-5%', 0.01, 0.05, (0.25, 0.45)),
       ('5-20%', 0.05, 0.20, (0.30, 0.50)), ('>20%', 0.20, float('inf'), (0.03, 0.10))]
for nm, lo, hi, (tlo, thi) in tgt:
    c = sum(1 for x in gaps if lo <= x < hi)
    okk = '✅' if tlo <= c / n <= thi else '❌'
    print(f'      {nm:<6s} {c / n:>6.1%}  目标 {tlo:.0%}-{thi:.0%}  {okk}')
sp = sorted(spreads)
print(f'    ③ 组内 spread 中位 {sp[len(sp) // 2]:.3f}  目标 0.30-0.60 ? '
      f'{"✅" if 0.30 <= sp[len(sp) // 2] <= 0.60 else "❌"}'
      f'   （p10 {sp[len(sp) // 10]:.3f}  p90 {sp[9 * len(sp) // 10]:.3f}）')
for thr in (0.10, 0.20, 0.30):
    c = sum(1 for x in sp if x > thr)
    print(f'       spread >{thr:.0%} 的组 {c}/{len(sp)} = {c / len(sp):.1%}'
          + ('   ← 交付方自评 groups_with_gt10pct_spread = 0.7133（差 2.5pp，'
             '疑为 spread 分母用 /min 而非 /median，**待复核不下定论**）' if thr == 0.10 else ''))
bg = sorted(best_gap)
print(f'    附：每组「次优 vs 最优」差 中位 {bg[len(bg) // 2]:.4f}'
      f'  ← 组内**最易**的那一步；若它都 <1%，该组的排序信号基本为零')
print(f'        metadata 自评 adjacent_diff_share = '
      f'{meta.get("group_delay", {}).get("adjacent_diff_share")}')
print(f'    宽松口径(步长>1%) {sum(1 for x in gaps if x > 0.01) / n:.1%}   '
      f'严格口径(>5%) {sum(1 for x in gaps if x > 0.05) / n:.1%}')

print()
print('完成（本脚本未修改任何文件）')
