"""DATA_SPEC_V2 合规性检查脚本（对照「数据规格说明书 V2（Rust 对齐版）」定稿）。

用法:
  python _check_v2_data.py [dir1 dir2 ...]
  不带参数时自动发现 data/batch*/、data/delivery*/batch*/ 下的
  circuit_static.parquet + timing_arcs.parquet（排除 archive*）。

检查项覆盖 V2 规格：I/O 任意 N/M、单 corner、vector=1、每电路行数公式、
组大小 10-15、完整性铁律（transistor_wave/gate_states/supply_noise/parasitic_caps
  列级 + 子字段级 100%）、DELAY 范围、JSON 列 key 集合一致性、
expr 不与 V1 重叠、sc_expansion 覆盖 + 命名一致、coverage_report.json。

输出: 逐项 [PASS/FAIL] + 末尾汇总。非致命问题用 [WARN]。
"""
import os, sys, json, glob, math
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CORNER = 's02p0_l01p0'          # V2 单 corner 标签
SLEW_S = 2e-12                  # 2 ps
LOAD_F = 1e-15                  # 1 fF
MIN_DELAY, MAX_DELAY = 1e-12, 1e-8

# transistor_wave_json 必含子字段（V2 方案 B）
TW_FIELDS = ['gate', 'ids_avg', 'ids_peak', 'vds_swing', 'ids_rise_time', 'vgs_swing', 'ids_charge']
# supply_noise_json 必含子字段（V2 方案 D）
SN_FIELDS = ['vdd_droop_mV', 'gnd_bounce_mV']

PASS, FAIL, WARN = [], [], []


def report(level, name, detail=''):
    tag = {'PASS': 'PASS', 'FAIL': 'FAIL', 'WARN': 'WARN'}[level]
    (PASS if level == 'PASS' else FAIL if level == 'FAIL' else WARN).append(name)
    print(f"[{tag}] {name}" + (f"  — {detail}" if detail else ""))


def load_json(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return '__INVALID_JSON__'
    return x


def parse_netlist_gates(nl):
    """从 gate_level_netlist 抽 (实例名 -> 门类型)。只认 X_ 行。"""
    gates = {}
    if not isinstance(nl, str):
        return gates
    for line in nl.splitlines():
        line = line.strip()
        if not line.startswith('X_'):
            continue
        toks = line.split()
        if len(toks) >= 2:
            gates[toks[0]] = toks[-1]
    return gates


def discover_pairs(extra_dirs):
    pairs = []
    for d in extra_dirs or []:
        sp, dp = os.path.join(d, 'circuit_static.parquet'), os.path.join(d, 'timing_arcs.parquet')
        if os.path.exists(sp) and os.path.exists(dp):
            pairs.append((d, [sp], [dp]))
        else:
            print(f"[WARN] 指定目录缺文件: {d}")
    if not pairs:
        # V1 目录（batch1/2/3, delivery*, archive*）不参与 V2 检查
        V1_NAMES = {'batch1', 'batch2', 'batch3', 'delivery1', 'delivery2', 'archive_v13.1'}
        for d in sorted(glob.glob('data/*')):
            base = os.path.basename(d)
            if not os.path.isdir(d) or base in V1_NAMES or base.startswith('archive'):
                continue
            sp, dp = os.path.join(d, 'circuit_static.parquet'), os.path.join(d, 'timing_arcs.parquet')
            if os.path.exists(sp) and os.path.exists(dp):
                pairs.append((d, [sp], [dp])); continue
            sparts = sorted(glob.glob(os.path.join(d, 'circuit_static_part*.parquet')))
            dparts = sorted(glob.glob(os.path.join(d, 'timing_arcs_part*.parquet')))
            if sparts and dparts:
                pairs.append((d, sparts, dparts))
    return pairs


def main():
    extra = sys.argv[1:]
    pairs = discover_pairs(extra)
    if not pairs:
        print("没有找到数据。用法: python _check_v2_data.py [data/batch4 data/batch5 ...]")
        sys.exit(1)
    print(f"发现 {len(pairs)} 个批次:\n  " + "\n  ".join(p[0] for p in pairs) + "\n")

    # ---------- 加载 ----------
    statics = [pd.read_parquet(p) for _, sps, _ in pairs for p in sps]
    dynamics = [pd.read_parquet(p) for _, _, dps in pairs for p in dps]
    st = pd.concat(statics, ignore_index=True)
    dy = pd.concat(dynamics, ignore_index=True)
    for c in ['candidate', 'candidate_id']:
        if c in st.columns:
            st = st.rename(columns={c: 'circuit_id'})
    st['circuit_id'] = st['circuit_id'].astype(str)
    dy['circuit_id'] = dy['circuit_id'].astype(str)
    # 批次间电路重叠（如 batch_v2 ⊂ batch_v2_full）：去重，保留第一个出现的
    n_dup_st = st.duplicated('circuit_id').sum()
    if n_dup_st:
        print(f"[WARN] static 中重复 circuit_id {n_dup_st} 行（批次重叠），已去重")
        kept = set(st['circuit_id'])
        st = st.drop_duplicates('circuit_id', keep='first')
        dy = dy[dy['circuit_id'].isin(kept)]  # 只去掉重复电路的行，保留每个电路全部样本行
    print(f"static 行数: {len(st)}   dynamic 行数: {len(dy)}  电路数: {st['circuit_id'].nunique()}\n")

    # ========== 一、静态表 circuit_static ==========
    print("===== 一、circuit_static.parquet =====")
    REQ_STATIC = ['circuit_id', 'expr', 'candidate_idx', 'transistor_count',
                  'gate_level_netlist', 'cell_types_json', 'input_pins_json',
                  'output_pins_json', 'pin_loads_json', 'parasitic_caps_json']
    missing = [c for c in REQ_STATIC if c not in st.columns]
    report('PASS' if not missing else 'FAIL', '静态必需列齐全',
           '' if not missing else f"缺: {missing}")

    # I/O 形状: 任意 N(1~16) / M(1~6)
    n_in = st['input_pins_json'].map(lambda x: len(load_json(x) or []))
    n_out = st['output_pins_json'].map(lambda x: len(load_json(x) or []))
    report('PASS' if (n_in >= 1).all() and (n_in <= 16).all() else 'FAIL',
           '输入引脚数 1~16', f"分布: {n_in.value_counts().sort_index().to_dict()}")
    report('PASS' if (n_out >= 1).all() and (n_out <= 6).all() else 'FAIL',
           '输出引脚数 1~6', f"分布: {n_out.value_counts().sort_index().to_dict()}")
    # vdd/gnd 不在引脚列表
    bad_pins = 0
    for _, r in st.iterrows():
        ip = load_json(r['input_pins_json']) or []
        op = load_json(r['output_pins_json']) or []
        if any(p in ('vdd', 'gnd') for p in ip + op):
            bad_pins += 1
    report('PASS' if bad_pins == 0 else 'FAIL', '电源/地不出现在 I/O 引脚列表', f"违规电路 {bad_pins}")

    # 引脚名 JSON 有效性
    bad_json = st['input_pins_json'].map(lambda x: load_json(x) is None or load_json(x) == '__INVALID_JSON__').sum() + \
               st['output_pins_json'].map(lambda x: load_json(x) is None or load_json(x) == '__INVALID_JSON__').sum()
    report('PASS' if bad_json == 0 else 'FAIL', 'I/O 引脚 JSON 全部有效', f"无效 {bad_json}")

    # circuit_id 格式 candidate_{expr}_{idx}
    bad_cid = 0
    for cid in st['circuit_id']:
        if not str(cid).startswith('candidate_'):
            bad_cid += 1
    report('PASS' if bad_cid == 0 else 'WARN', 'circuit_id 格式 candidate_*', f"不合规 {bad_cid}")

    # transistor_count
    report('PASS' if (st['transistor_count'] > 0).all() else 'FAIL',
           'transistor_count 全部 > 0', f"min={st['transistor_count'].min()}")

    # 网表格式: .SUBCKT DUT + .ENDS DUT + X_ 行存在
    nl_ok = st['gate_level_netlist'].map(lambda s: isinstance(s, str) and '.SUBCKT' in s and '.ENDS' in s).sum()
    report('PASS' if nl_ok == len(st) else 'FAIL', '网表含 .SUBCKT/.ENDS', f"{nl_ok}/{len(st)}")
    x_rows = st['gate_level_netlist'].map(lambda s: sum(1 for l in (s or '').splitlines() if l.strip().startswith('X_'))).sum()
    report('PASS' if x_rows > 0 else 'FAIL', '网表含 X_ 实例行', f"总 X_ 行 {x_rows}")

    # cell_types_json 与网表门名一致
    ct_mismatch = 0; n_checked = 0
    for _, r in st.iterrows():
        ct = load_json(r['cell_types_json'])
        gates = parse_netlist_gates(r['gate_level_netlist'])
        if not isinstance(ct, list) or not gates:
            ct_mismatch += 1; continue
        n_checked += 1
        if set(ct) != set(gates.values()):
            ct_mismatch += 1
    report('PASS' if ct_mismatch == 0 else 'FAIL',
           'cell_types_json 与网表门名一致', f"不一致 {ct_mismatch}/{len(st)} (检查 {n_checked})")

    # parasitic_caps_json: 列非空 100% + key 集合 = 门实例 + in_*/out + 值>0
    pc_nonnull = st['parasitic_caps_json'].notna().sum()
    report('PASS' if pc_nonnull == len(st) else 'FAIL', 'parasitic_caps_json 列非空 100%',
           f"{pc_nonnull}/{len(st)}")
    pc_key_ok = 0; pc_sub_ok = 0; pc_val_ok = 0; pc_n = 0
    for _, r in st.sample(min(3000, len(st))).iterrows():
        pc = load_json(r['parasitic_caps_json'])
        gates = parse_netlist_gates(r['gate_level_netlist'])
        if not isinstance(pc, dict) or not gates:
            continue
        pc_n += 1
        if set(pc.keys()) == set(gates.keys()):
            pc_key_ok += 1
        good = True
        for inst, sub in pc.items():
            if not isinstance(sub, dict) or 'out' not in sub or not any(k.startswith('in_') for k in sub):
                good = False; break
            for k, v in sub.items():
                if not isinstance(v, (int, float)) or not (v > 0):
                    good = False; break
        if good:
            pc_sub_ok += 1
    report('PASS' if pc_n > 0 and pc_key_ok == pc_n else 'WARN', 'parasitic_caps key=门实例(抽样)',
           f"{pc_key_ok}/{pc_n} (n={pc_n})")
    report('PASS' if pc_n > 0 and pc_sub_ok == pc_n else 'FAIL',
           'parasitic_caps 子字段 in_*/out 且 >0(抽样)',
           f"{pc_sub_ok}/{pc_n} (n={pc_n})  — 缺 in_* 或值≤0 的电路")

    # pin_loads_json 覆盖所有输入+输出
    pl_ok = 0; pl_n = 0
    for _, r in st.sample(min(3000, len(st))).iterrows():
        pl = load_json(r['pin_loads_json'])
        ip = load_json(r['input_pins_json']) or []
        op = load_json(r['output_pins_json']) or []
        if not isinstance(pl, dict):
            continue
        pl_n += 1
        if all(p in pl for p in ip + op):
            pl_ok += 1
    report('PASS' if pl_n > 0 and pl_ok == pl_n else 'WARN', 'pin_loads_json 覆盖全部 I/O 引脚(抽样)',
           f"{pl_ok}/{pl_n} (n={pl_n})")

    # ========== 二、动态表 timing_arcs ==========
    print("\n===== 二、timing_arcs.parquet =====")
    REQ_DYN = ['circuit_id', 'corner', 'switching_pin', 'direction', 'expr', 'candidate_idx',
               'vector', 'slew_s', 'output_load_f', 'DELAY', 'pin_slew_json', 'pin_load_json',
               'gate_states_json', 'transistor_wave_json', 'supply_noise_json']
    missing = [c for c in REQ_DYN if c not in dy.columns]
    report('PASS' if not missing else 'FAIL', '动态必需列齐全',
           '' if not missing else f"缺: {missing}")
    if 'corner' not in dy.columns:
        report('FAIL', 'corner 列存在')
    else:
        corners = dy['corner'].astype(str).value_counts().to_dict()
        report('PASS' if set(corners) <= {CORNER} else 'FAIL', '单 corner = s02p0_l01p0',
               f"{corners}")
    # per_gate_timing_json 已废弃不应出现
    report('PASS' if 'per_gate_timing_json' not in dy.columns else 'FAIL',
           'per_gate_timing_json 已废弃(不应存在)')
    # direction 取值（兼容 'rise.sp'/'fall.sp' 后缀，分别报告）
    if 'direction' in dy.columns:
        raw_dir = set(dy['direction'].astype(str).unique())
        stripped = set(dy['direction'].astype(str).str.replace('.sp', '').unique())
        dvals = set(dy['direction'].astype(str).str.replace('.sp', '').unique())
        report('PASS' if dvals <= {'rise', 'fall'} else 'FAIL',
               'direction ∈ {rise, fall}(去 .sp 后缀后)',
               f"原始值: {raw_dir} → 归一后: {dvals}")
        if raw_dir - {'rise', 'fall'}:
            report('WARN', 'direction 含 .sp 后缀(应为纯 rise/fall)',
                   f"{raw_dir - {'rise', 'fall'}}")
    else:
        report('FAIL', 'direction 列存在')
    # DELAY 范围
    d = dy['DELAY']
    report('PASS' if (d > MIN_DELAY).all() and (d < MAX_DELAY).all() else 'FAIL',
           f'DELAY 范围 ({MIN_DELAY:.0e}, {MAX_DELAY:.0e})',
           f"min={d.min():.3e} max={d.max():.3e}")
    # slew_s / output_load_f > 0
    report('PASS' if (dy['slew_s'] > 0).all() else 'FAIL', 'slew_s 全部 > 0')
    report('PASS' if (dy['output_load_f'] > 0).all() else 'FAIL', 'output_load_f 全部 > 0')

    # switching_pin ∈ input_pins_json
    pinmap = dict(zip(st['circuit_id'], st['input_pins_json'].map(load_json)))
    bad_sw = 0; n_sw = 0
    for cid, pin in zip(dy['circuit_id'], dy['switching_pin']):
        pins = pinmap.get(cid)
        n_sw += 1
        if not pins or str(pin) not in pins:
            bad_sw += 1
    report('PASS' if bad_sw == 0 else 'FAIL', 'switching_pin ∈ input_pins_json', f"{bad_sw}/{n_sw}")

    # vector: 长度 = N_in；位与 direction 一致（rise=0, fall=1；direction 去 .sp 后缀）
    vlen_ok = 0; vbit_ok = 0; v_n = 0; sw_bit1 = 0
    for _, r in dy.head(20000).iterrows():
        pins = pinmap.get(r['circuit_id'])
        vec = r['vector']
        if not pins or not isinstance(vec, str):
            continue
        v_n += 1
        if len(vec) == len(pins):
            vlen_ok += 1
            try:
                idx = list(pins).index(str(r['switching_pin']))
                dirn = str(r['direction']).replace('.sp', '')
                expect = '0' if dirn == 'rise' else '1'
                if vec[idx] == expect:
                    vbit_ok += 1
                if vec[idx] == '1':
                    sw_bit1 += 1
            except ValueError:
                pass
    report('PASS' if v_n > 0 and vlen_ok == v_n else 'WARN', 'vector 长度 = 输入数(前2万行)',
           f"{vlen_ok}/{v_n}")
    report('PASS' if v_n > 0 and vbit_ok == v_n else 'WARN',
           'vector 切换位与 direction 一致(前2万行, 去.sp)',
           f"{vbit_ok}/{v_n} (切换位=1 占比 {sw_bit1}/{v_n} —— 若≈100% 说明 rise 行也是 1, 不合规)")

    # 每电路行数 = 2 × N_in × M
    rowcnt = dy.groupby('circuit_id').size()
    expect_rows = 2 * n_in * n_out
    expect_rows = expect_rows.set_axis(st['circuit_id'])  # 对齐
    merged = pd.DataFrame({'actual': rowcnt, 'expect': expect_rows}).dropna()
    mism = merged[merged['actual'] != merged['expect']]
    report('PASS' if len(mism) == 0 else 'FAIL', '每电路行数 = 2×N_in×M',
           f"不匹配 {len(mism)}/{len(merged)} (例如: {mism.head(3).to_dict('index') if len(mism) else ''})")

    # vector=1: (circuit, corner, switching_pin, direction) 唯一 vector
    if 'corner' in dy.columns:
        gk = ['circuit_id', 'corner', 'switching_pin', 'direction']
    else:
        gk = ['circuit_id', 'switching_pin', 'direction']
    nvec = dy.groupby(gk)['vector'].nunique()
    report('PASS' if (nvec == 1).all() else 'FAIL', '每 (circuit,pin,dir) 恰 1 个 vector',
           f"违规组 {(nvec != 1).sum()}/{len(nvec)}")

    # 无重复行 (circuit_id, corner, switching_pin, direction, vector)
    dup = dy.duplicated(subset=['circuit_id', 'corner', 'switching_pin', 'direction', 'vector']).sum() \
        if 'corner' in dy.columns else dy.duplicated(subset=['circuit_id', 'switching_pin', 'direction', 'vector']).sum()
    report('PASS' if dup == 0 else 'FAIL', '无重复样本行', f"重复 {dup}")

    # gate_states_json: 列非空 100% + key 集合 = 门实例(抽样)
    gs_nonnull = dy['gate_states_json'].notna().sum()
    report('PASS' if gs_nonnull == len(dy) else 'FAIL', 'gate_states_json 列非空 100%', f"{gs_nonnull}/{len(dy)}")
    nlmap = dict(zip(st['circuit_id'], st['gate_level_netlist']))
    gs_ok = 0; gs_n = 0
    for _, r in dy.sample(min(3000, len(dy))).iterrows():
        gs = load_json(r['gate_states_json'])
        gates = parse_netlist_gates(nlmap.get(r['circuit_id'], ''))
        if not isinstance(gs, dict) or not gates:
            continue
        gs_n += 1
        if set(gs.keys()) == set(gates.keys()):
            gs_ok += 1
    report('PASS' if gs_n > 0 and gs_ok == gs_n else 'WARN', 'gate_states key=门实例(抽样)', f"{gs_ok}/{gs_n}")

    # 大小写一致性诊断（gate_states 用动态表 / parasitic_caps 用静态表 / transistor_wave.gate vs 网表 X_*）
    cs_pc = 0; cs_n = 0
    for _, r in st.sample(min(3000, len(st))).iterrows():
        gates = parse_netlist_gates(r['gate_level_netlist'])
        pc = load_json(r['parasitic_caps_json'])
        if not gates:
            continue
        cs_n += 1
        pc_keys = set(pc.keys()) if isinstance(pc, dict) else set()
        if pc_keys and pc_keys == set(gates.keys()):
            cs_pc += 1
    cs_gs = 0; gs_cs_n = 0
    for _, r in dy.sample(min(3000, len(dy))).iterrows():
        gates = parse_netlist_gates(nlmap.get(r['circuit_id'], ''))
        gs = load_json(r['gate_states_json'])
        if not gates or not isinstance(gs, dict):
            continue
        gs_cs_n += 1
        if set(gs.keys()) == set(gates.keys()):
            cs_gs += 1
    cs_tw = 0
    for _, r in dy.sample(min(2000, len(dy))).iterrows():
        tw = load_json(r['transistor_wave_json'])
        if isinstance(tw, dict) and len(tw) > 0:
            gate_vals = {sub.get('gate') for sub in tw.values() if isinstance(sub, dict)}
            if gate_vals and all(isinstance(g, str) and g.startswith('X_') for g in gate_vals):
                cs_tw += 1
    if cs_n:
        report('PASS' if cs_pc == cs_n else 'FAIL',
               'parasitic_caps key 与网表 X_* 完全一致(大小写)(抽样)',
               f"{cs_pc}/{cs_n}")
    report('PASS' if gs_cs_n > 0 and cs_gs == gs_cs_n else 'FAIL',
           'gate_states key 与网表 X_* 完全一致(大小写)(抽样, 动态表)',
           f"{cs_gs}/{gs_cs_n} (小写 x_* 会被 graph_builder 查不到)")
    report('PASS' if cs_tw == 2000 else 'WARN',
           'transistor_wave gate 字段为 X_* 大写(抽样)', f"{cs_tw}/2000")

    # transistor_wave_json: 列非空 100% + 子字段齐全 + 值域(抽样)
    tw_nonnull = dy['transistor_wave_json'].notna().sum()
    report('PASS' if tw_nonnull == len(dy) else 'FAIL', 'transistor_wave_json 列非空 100%', f"{tw_nonnull}/{len(dy)}")
    tw_field_ok = 0; tw_val_ok = 0; tw_n = 0; tw_empty = 0; tw_charge_copy = 0; tw_both_zero = 0; tw_active = 0
    for _, r in dy.sample(min(5000, len(dy))).iterrows():
        tw = load_json(r['transistor_wave_json'])
        if not isinstance(tw, dict) or len(tw) == 0:
            tw_empty += 1; continue
        tw_n += 1
        fields_good = all(isinstance(tw[m], dict) and all(f in tw[m] for f in TW_FIELDS) for m in tw)
        vals_good = True
        if fields_good:
            for m, sub in tw.items():
                for f in ['ids_avg', 'ids_peak', 'ids_rise_time', 'ids_charge']:
                    if not isinstance(sub[f], (int, float)) or sub[f] < 0:
                        vals_good = False; break
                for f in ['vds_swing', 'vgs_swing']:
                    if not isinstance(sub[f], (int, float)) or sub[f] < 0:
                        vals_good = False; break
                if not isinstance(sub['gate'], str) or not sub['gate'].startswith('X_'):
                    vals_good = False; break
                if isinstance(sub['ids_charge'], (int, float)) and isinstance(sub['ids_avg'], (int, float)):
                    if sub['ids_avg'] == 0 and sub['ids_charge'] == 0:
                        tw_both_zero += 1            # 非翻转管 0==0 合法
                    else:
                        tw_active += 1
                        if abs(sub['ids_charge'] - sub['ids_avg']) < 1e-9:
                            tw_charge_copy += 1      # 激活管仍等于 ids_avg = 复制残留
        if fields_good:
            tw_field_ok += 1
        if fields_good and vals_good:
            tw_val_ok += 1
    report('PASS' if tw_n > 0 and tw_empty == 0 else 'FAIL', 'transistor_wave 非空 JSON(抽样)',
           f"空/无效 {tw_empty}/{tw_n + tw_empty}")
    report('PASS' if tw_n > 0 and tw_field_ok == tw_n else 'WARN', 'transistor_wave 7 子字段齐全(抽样)',
           f"{tw_field_ok}/{tw_n}")
    report('PASS' if tw_n > 0 and tw_val_ok == tw_n else 'WARN', 'transistor_wave 值域合规(抽样)',
           f"{tw_val_ok}/{tw_n}")
    report('PASS' if tw_charge_copy == 0 else ('WARN' if tw_charge_copy / max(tw_active, 1) < 0.01 else 'FAIL'),
           'ids_charge ≠ ids_avg(激活管; 0==0 非翻转管不计)',
           f"复制残留 {tw_charge_copy}/{tw_active} 激活管 (非翻转 0==0 {tw_both_zero} 个正常); "
           f"<1% 记 WARN, ≥1% 记 FAIL")

    # supply_noise_json: 列非空 100% + 子字段 + ≥0(抽样)
    sn_nonnull = dy['supply_noise_json'].notna().sum()
    report('PASS' if sn_nonnull == len(dy) else 'FAIL', 'supply_noise_json 列非空 100%', f"{sn_nonnull}/{len(dy)}")
    sn_ok = 0; sn_n = 0
    for _, r in dy.sample(min(3000, len(dy))).iterrows():
        sn = load_json(r['supply_noise_json'])
        if not isinstance(sn, dict) or not all(f in sn for f in SN_FIELDS):
            continue
        if all(isinstance(sn[f], (int, float)) and sn[f] >= 0 for f in SN_FIELDS):
            sn_ok += 1
        sn_n += 1
    report('PASS' if sn_n > 0 and sn_ok == sn_n else 'WARN', 'supply_noise 子字段 + ≥0(抽样)', f"{sn_ok}/{sn_n}")

    # pin_slew_json / pin_load_json: key = input_pins_json；切换 pin slew = slew_s；非切换 = 0(抽样)
    slew_ok = 0; slew_n = 0
    for _, r in dy.sample(min(3000, len(dy))).iterrows():
        ps = load_json(r['pin_slew_json'])
        pins = pinmap.get(r['circuit_id'])
        if not isinstance(ps, dict) or not pins or set(ps.keys()) != set(pins):
            continue
        slew_n += 1
        good = True
        for p in pins:
            if p == str(r['switching_pin']):
                if abs(ps[p] - r['slew_s']) > 1e-15 * max(1, abs(r['slew_s'])):
                    good = False
            elif ps[p] != 0:
                good = False
        if good:
            slew_ok += 1
    report('PASS' if slew_n > 0 and slew_ok == slew_n else 'WARN',
           'pin_slew_json key=输入 & 切换=slew_s & 非切换=0(抽样)', f"{slew_ok}/{slew_n}")
    pl2_ok = 0; pl2_n = 0
    for _, r in dy.sample(min(3000, len(dy))).iterrows():
        pl = load_json(r['pin_load_json'])
        pins = pinmap.get(r['circuit_id'])
        if not isinstance(pl, dict) or not pins or set(pl.keys()) != set(pins):
            continue
        pl2_n += 1
        if all(pl[p] >= 0 for p in pins):
            pl2_ok += 1
    report('PASS' if pl2_n > 0 and pl2_ok == pl2_n else 'WARN',
           'pin_load_json key=输入 & 非负(抽样)', f"{pl2_ok}/{pl2_n}")

    # ========== 三、批次/组规模 ==========
    print("\n===== 三、组规模 & 总量 =====")
    grp = st.groupby('expr')['circuit_id'].nunique()
    in_range = ((grp >= 10) & (grp <= 15)).mean() * 100
    report('PASS' if in_range >= 95 else 'WARN', '每组变体数 10~15',
           f"中位 {grp.median():.0f} 范围 [{grp.min()},{grp.max()}] 合规 {in_range:.1f}%")
    print(f"  expr 组数: {len(grp)}   电路数: {len(st)}   dynamic 行数: {len(dy)}")
    # I/O 分桶（信息）
    bins_in = pd.cut(n_in, [0, 2, 4, 8, 16], labels=['1~2', '3~4', '5~8', '9~16'])
    print(f"  输入分桶: {bins_in.value_counts().to_dict()}")
    print(f"  输出分布: {n_out.value_counts().sort_index().to_dict()}")

    # expr 不与 V1 重叠（本地 delivery1+2 作参照）
    v1_exprs = set()
    for d in ['data/delivery1', 'data/delivery2']:
        p = os.path.join(d, 'batch1', 'circuit_static.parquet')
        if os.path.exists(p):
            v1_exprs |= set(pd.read_parquet(p, columns=['expr'])['expr'].astype(str))
        for b in ['batch2', 'batch3']:
            p = os.path.join(d, b, 'circuit_static.parquet')
            if os.path.exists(p):
                v1_exprs |= set(pd.read_parquet(p, columns=['expr'])['expr'].astype(str))
            for part in glob.glob(os.path.join(d, b, 'circuit_static_part*.parquet')):
                v1_exprs |= set(pd.read_parquet(part, columns=['expr'])['expr'].astype(str))
    cur_exprs = set(st['expr'].astype(str))
    overlap = cur_exprs & v1_exprs
    report('PASS' if not overlap else 'FAIL', 'expr 不与 V1(569) 重叠',
           f"重叠 {len(overlap)} 个: {sorted(overlap)[:5] if overlap else ''}")

    # 批次间电路重叠（batch_v2 vs batch_v2_full 等）
    cid_by_batch = {}
    for d, sps, _ in pairs:
        cids = set()
        for p in sps:
            cids |= set(pd.read_parquet(p, columns=['circuit_id'])['circuit_id'].astype(str))
        cid_by_batch[os.path.basename(d)] = cids
    dup_cids = set()
    for i, (n1, s1) in enumerate(cid_by_batch.items()):
        for n2, s2 in list(cid_by_batch.items())[i + 1:]:
            ov = s1 & s2
            if ov:
                dup_cids |= ov
                print(f"  [WARN] {n1} ∩ {n2} = {len(ov)} 个 circuit_id 重叠")
    report('PASS' if not dup_cids else 'WARN', '批次间无电路重叠', f"重叠 {len(dup_cids)} 个")

    # sc_expansion 覆盖 SC_ cell 名 + 命名一致
    if os.path.exists('data/sc_expansion.json'):
        with open('data/sc_expansion.json', encoding='utf-8') as f:
            sc = json.load(f)
        cells = set()
        for c in st['cell_types_json']:
            v = load_json(c)
            if isinstance(v, list):
                cells |= {x for x in v if isinstance(x, str)}
        sc_cells = {c for c in cells if c.startswith('SC_')}
        missing_sc = {c for c in sc_cells if c not in sc}
        bad_exp = {c for c in sc_cells if c in sc and (not isinstance(sc[c], dict) or not sc[c].get('subcircuit'))}
        report('PASS' if not missing_sc and not bad_exp else 'FAIL',
               'sc_expansion.json 覆盖全部 SC_ 名(命名一致铁律)',
               f"SC_ 名 {len(sc_cells)} 个, 缺 {len(missing_sc)}: {sorted(missing_sc)[:5] if missing_sc else ''}, 展开为空 {len(bad_exp)}")
    else:
        report('WARN', 'sc_expansion.json 不存在(未检查)')

    # coverage_report.json（V2 交付名称为 coverage_report_v2.json）
    cov_path = None
    for cand in ['data/coverage_report_v2.json', 'data/coverage_report.json']:
        if os.path.exists(cand):
            cov_path = cand
            break
    if cov_path:
        with open(cov_path, encoding='utf-8') as f:
            cov = json.load(f)
        report('PASS', f'coverage_report 存在 ({cov_path})')
        print(f"  coverage_report keys: {list(cov.keys())}")
        n_bad = 0
        for bk, bv in cov.items():
            if isinstance(bv, dict) and 'fields' in bv:
                for fn, fs in bv['fields'].items():
                    if isinstance(fs, str) and '100%' not in fs:
                        n_bad += 1
                        print(f"    [WARN] {bk}.{fn}: {fs}")
        report('PASS' if n_bad == 0 else 'WARN', 'coverage_report 全部 100%', f"非100% {n_bad} 项")
    else:
        report('FAIL', 'coverage_report 存在（V2 要求随数据交付）')

    # ========== 汇总 ==========
    print("\n===== 汇总 =====")
    print(f"PASS {len(PASS)} | FAIL {len(FAIL)} | WARN {len(WARN)}")
    if FAIL:
        print("FAIL 项:")
        for n in FAIL:
            print(f"  ✗ {n}")
    if WARN:
        print("WARN 项（建议人工复核）:")
        for n in WARN:
            print(f"  ~ {n}")
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
