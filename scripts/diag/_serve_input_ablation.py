"""_serve_input_ablation.py — serve 侧「输入退化」代价消融（离线、不动训练数据）。

背景（2026-09-15 排查「GNN 与 Rust 部署不同步」）：
serve.py 喂给模型的三处输入与训练侧不一致，且三处都是**静默**的（serve 不报错、_smoke_v2.py
用的是真实 dataset 对象所以永远测不出）：

  ① circuit_sig[1]（晶体管数）
       训练：src/data_loader.py:320-325  [n_gates, static_df.transistor_count, #input_pins]
       serve：scripts/diag/serve.py:281  [num_nodes, **0.0 硬编码**, len(pins)]
  ② struct_prior
       训练：data_loader.py:710-719  [transistor_count, #SC_AND, #SC_INV_WIRE]（USE_STRUCT_PRIOR=True）
       serve：serve.py:286 传 None → src/model.py:110 `if struct_prior is not None` 为 False
              → **整条结构残差被静默跳过**
  ③ 动态 load（动态 7 维里的 feat[3]）
       训练：data_loader.py:392-398 逐行读 pin_load_json（实测中位 1.2e-15，范围 3e-16~4.5e-15）
       serve：serve.py:117 `load = global_load`（常数 LOAD_F=1e-15，落在训练分布内 → 信号被抹平）

①② 都进**未归一化**的 MLP（src/model.py:39-54 的 sig_encoder / struct_encoder 直接 Linear(3,·)），
量级 52~316；③ 进的是已 scaler 归一化的连续块。

本脚本：在同一批测试电路上，用同一个 checkpoint，**只改这三处输入**，量出各自代价与可修部分的收益。
口径与训练/部署一致：preds = 10**out（线性），targets = DELAY，ranking_metrics(..., avg_delay=True)。
所有 x 的修改只动「动态 7 维里 load 那一列」和「图级 csig / prior 两个张量」，其余（含 gate_states、
logic、vector、静态块、edge_index）逐字节保持 dataset 给出的值 → 差值可干净归因。

用法（server，项目根目录；run 目录约定 ~/project-107-<variant>）：
  OMP_NUM_THREADS=6 ~/venv/bin/python3 scripts/diag/_serve_input_ablation.py \
      --ckpt ~/project-107-v2nowave42m4/outputs/midpoint_ep250.pt \
      --scaler ~/project-107-v2nowave42m4/outputs/scaler.pkl --out-json /tmp/abl.json

  # 全量测试集（慢）：
  ... --max-exprs 0
"""
import sys, os, json, argparse, glob, time

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import torch

# ---- 先 import serve：它内部就把 config 强制成 no-wave 推理态（STRUCT_MODE 可 env 覆盖）----
import scripts.diag.serve as S
import config

# serve 侧常量（与 serve.py:54-56 同源，直接引用保证不漂移）
SLEW_S = S.SLEW_S
LOAD_F = S.LOAD_F
CORNER = S.CORNER

from src.data_loader import DelayDataset
from src.utils import split_by_expr, ranking_metrics, load_scaler
from src.graph_builder import rebuild_gate_types
import src.graph_builder as gb


# ---------------------------------------------------------------- 数据侧（严格复刻 train_sweep.main）
def build_test_frame(batches, min_group_size, split_seed):
    static_parquets, dynamic_parquets = [], []
    for b in batches:
        b = b.strip()
        if not b:
            continue
        sp = os.path.join(PROJ, f'data/{b}/circuit_static.parquet')
        dp = os.path.join(PROJ, f'data/{b}/timing_arcs.parquet')
        if os.path.exists(sp) and os.path.exists(dp):
            static_parquets.append(sp); dynamic_parquets.append(dp); continue
        sparts = sorted(glob.glob(os.path.join(PROJ, f'data/{b}/circuit_static_part*.parquet')))
        dparts = sorted(glob.glob(os.path.join(PROJ, f'data/{b}/timing_arcs_part*.parquet')))
        if sparts and dparts:
            static_parquets.extend(sparts); dynamic_parquets.extend(dparts)
    if not static_parquets:
        raise SystemExit(f'[abl] 未找到任何批次数据: {batches}（需在项目根目录跑）')

    # 动态：wave OFF 时排除 transistor_wave_json（省内存，train_sweep 同款）
    import pyarrow.parquet as _pq
    dyn_dfs = []
    for p in dynamic_parquets:
        cols = [c for c in _pq.read_schema(p).names if c != 'transistor_wave_json']
        dyn_dfs.append(pd.read_parquet(p, columns=cols))
    dyn = pd.concat(dyn_dfs, ignore_index=True)
    for c in ['candidate', 'candidate_id']:
        if c in dyn.columns and 'circuit_id' not in dyn.columns:
            dyn = dyn.rename(columns={c: 'circuit_id'})
    if 'delay_s' in dyn.columns and 'DELAY' not in dyn.columns:
        dyn = dyn.rename(columns={'delay_s': 'DELAY'})
    n0 = len(dyn)
    dyn = dyn.dropna(subset=['circuit_id', 'DELAY'])
    dyn['circuit_id'] = dyn['circuit_id'].astype(str)
    dyn = dyn[(dyn['DELAY'] > 1e-12) & (dyn['DELAY'] < 1e-8)].reset_index(drop=True)
    print(f'[abl] 动态清洗: {n0} -> {len(dyn)} 行, {dyn["circuit_id"].nunique()} 电路')
    if min_group_size > 1 and 'expr' in dyn.columns:
        gs = dyn.groupby('expr')['circuit_id'].nunique()
        keep = gs[gs >= min_group_size].index.astype(str)
        dyn = dyn[dyn['expr'].astype(str).isin(keep)].reset_index(drop=True)
        print(f'[abl] 组过滤(>= {min_group_size}): -> {len(dyn)} 行')

    ids = dyn['circuit_id'].unique().tolist()
    id2e = dict(zip(dyn['circuit_id'].astype(str), dyn['expr'].astype(str))) \
        if 'expr' in dyn.columns else None
    train_ids, val_ids, test_ids = split_by_expr(ids, id2e, seed=split_seed)
    print(f'[abl] 切分(seed={split_seed}): train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} 电路')
    return static_parquets, dynamic_parquets, dyn, test_ids


def rebuild_vocab(static_parquets):
    """train_sweep.py:459-471 的同款调用。注意 src/graph_builder.rebuild_gate_types 现已忽略入参，
    恒为 LOGIC_TYPES + 3 reserved（logic_only 口径），所以词表与 serve 天然一致；
    仍然照抄这一步是为了与训练 runbook 的调用顺序逐条对齐（自检用）。"""
    all_types = set()
    for p in static_parquets:
        df = pd.read_parquet(p)
        for col in ['candidate', 'candidate_id']:
            if col in df.columns and 'circuit_id' not in df.columns:
                df = df.rename(columns={col: 'circuit_id'})
        if 'cell_types_json' not in df.columns:
            continue
        for v in df['cell_types_json']:
            try:
                all_types.update(json.loads(v) if isinstance(v, str) else v)
            except Exception:
                pass
    rebuild_gate_types(all_types)
    print(f'[abl] gate 词表重建: {len(all_types)} 类 -> GATE_TYPES={len(gb.GATE_TYPES)}')


class RecDelayDataset(DelayDataset):
    """记录最近一次 __getitem__ 里哪些节点拿到了动态特征（= 输入引脚节点）→ 用于定位 load 列。"""

    def _get_dynamic_features(self, row, pin_loads_dict, pins=None):
        dyn, corner = super()._get_dynamic_features(row, pin_loads_dict, pins)
        self.last_dyn_keys = set(dyn.keys())
        return dyn, corner


# ------------------------------------------------------- serve 侧「够不够得着」的判定
def _cells_of(netlist):
    out = []
    for l in str(netlist or '').split('\n'):
        s = l.strip()
        if s.startswith('X_') and len(s.split()) >= 3:
            out.append(s.split()[-1])
    return out


def probe_prior_tc(static_parquets, n_sample=200):
    """回答两个决定「能不能在 serve 侧修」的前置问题：
      (1) cell_types_json 是「唯一类型表」还是「逐实例表」→ 决定训练侧 #SC_AND / #SC_INV_WIRE 的口径；
      (2) sc_expansion.json 对 V2 生成器 cell 名的覆盖率 → 决定 Σ n_t（= 晶体管数）能不能从网表复算。
    serve 侧两个候选修法都建立在「网表里已经有足够信息」之上，这里先把这个前提验掉。"""
    sc = gb._load_sc_expansion()
    st = pd.concat([pd.read_parquet(p) for p in static_parquets], ignore_index=True)
    for c in ['candidate', 'candidate_id']:
        if c in st.columns and 'circuit_id' not in st.columns:
            st = st.rename(columns={c: 'circuit_id'})
    st['circuit_id'] = st['circuit_id'].astype(str)
    st = st.drop_duplicates('circuit_id').reset_index(drop=True)
    if len(st) > n_sample:
        st = st.iloc[np.linspace(0, len(st) - 1, n_sample).astype(int)].reset_index(drop=True)

    ok_ct_uniq = ok_ct_full = 0
    covs, tc_err_all, tc_err_uniq = [], [], []
    d_and, d_inv, d_and_lc, d_inv_lc = [], [], [], []
    for _, r in st.iterrows():
        cells = _cells_of(r.get('gate_level_netlist', ''))
        try:
            ct = json.loads(r['cell_types_json']) if isinstance(r['cell_types_json'], str) else r['cell_types_json']
        except Exception:
            ct = None
        ct = ct if isinstance(ct, list) else []
        if cells and ct:
            if len(ct) == len(set(ct)):
                ok_ct_uniq += 1
            if len(ct) == len(cells):
                ok_ct_full += 1
        covs.append(sum(1 for c in set(cells) if c in sc) / max(len(set(cells)), 1))
        tc_meta = float(r.get('transistor_count', 0) or 0)
        if tc_meta > 0:
            tc_a = sum(float(gb.gate_struct(c)['n_t']) for c in cells)
            tc_u = sum(float(gb.gate_struct(c)['n_t']) for c in set(cells))
            tc_err_all.append(abs(tc_a - tc_meta) / tc_meta)
            tc_err_uniq.append(abs(tc_u - tc_meta) / tc_meta)
        d_and.append(sum(1 for g in ct if 'SC_AND' in str(g) and 'SC_AND_' not in str(g)))
        d_inv.append(sum(1 for g in ct if 'SC_INV_WIRE' in str(g)))
        d_and_lc.append(sum(1 for c in cells if gb.gate_struct(c)['logic'] == 'AND'))
        d_inv_lc.append(sum(1 for c in cells if gb.gate_struct(c)['logic'] in ('INV', 'BUF')))

    n = len(st)
    print('\n' + '-' * 118)
    print(f'[probe] serve 侧可复算性判定（{n} 个电路抽样）')
    print('-' * 118)
    print(f'  cell_types_json: len==len(set) 的占比 {ok_ct_uniq/n*100:.1f}%  '
          f'len==#X_ 实例数的占比 {ok_ct_full/n*100:.1f}%   '
          f'（前者 100% → 唯一类型表；后者 100% → 逐实例表）')
    print(f'  sc_expansion 覆盖率（唯一 cell 名命中率）: 中位 {np.median(covs)*100:.1f}%  '
          f'最小 {np.min(covs)*100:.1f}%  =100% 的电路占比 {np.mean([c>=0.999 for c in covs])*100:.1f}%')
    if tc_err_all:
        print(f'  Σ n_t（逐实例）vs transistor_count: 中位相对误差 {np.median(tc_err_all)*100:.1f}%  '
              f'误差<1% 的占比 {np.mean([e<0.01 for e in tc_err_all])*100:.1f}%')
        print(f'  Σ n_t（唯一类型）vs transistor_count: 中位相对误差 {np.median(tc_err_uniq)*100:.1f}%  '
              f'误差<1% 的占比 {np.mean([e<0.01 for e in tc_err_uniq])*100:.1f}%')
    da, di, dal, dil = (np.array(x, float) for x in (d_and, d_inv, d_and_lc, d_inv_lc))
    print(f'  #SC_AND 训练口径中位 {np.median(da):.1f} ; 逻辑类=AND 复算中位 {np.median(dal):.1f} ; '
          f'完全相等占比 {np.mean(da == dal)*100:.1f}%  相关 {np.corrcoef(da, dal)[0,1] if da.std() and dal.std() else float("nan"):.3f}')
    print(f'  #SC_INV_WIRE 训练口径中位 {np.median(di):.1f} ; 逻辑类∈(INV,BUF) 复算中位 {np.median(dil):.1f} ; '
          f'完全相等占比 {np.mean(di == dil)*100:.1f}%  相关 {np.corrcoef(di, dil)[0,1] if di.std() and dil.std() else float("nan"):.3f}')
    print('-' * 118 + '\n')


def verify_npz(npz_path, targets_full):
    """免费闸门：arm 目录里的 test_predictions.npz 存的是该 arm 自己 test_dataset 行序的 targets。
    与本脚本重建的全量 test_df 逐位比对 → 一次性验掉「批次集合 / 清洗过滤 / MIN_GROUP_SIZE / split_seed /
    行序」这一整串假设。通过 = 绝对指标可与该 arm 的历史数字对读；不通过 = 只有配对差值有效。"""
    if not npz_path or not os.path.exists(npz_path):
        print(f'[gate] 未找到 npz（{npz_path}）→ 跳过切分自证；绝对数按「重建切分」理解')
        return None
    try:
        d = np.load(npz_path)
    except Exception as e:
        print(f'[gate] npz 读取失败 ({e}) → 跳过')
        return None
    tg = d['targets'] if 'targets' in d.files else None
    print(f'[gate] npz: preds={d["preds"].shape}  '
          f'targets={None if tg is None else tg.shape}  本脚本重建全量 test 行数={len(targets_full)}')
    if tg is None:
        return d
    if len(tg) != len(targets_full):
        print(f'[gate] ❌ 行数不一致（npz {len(tg)} vs 重建 {len(targets_full)}）→ 切分/批次/过滤与训练不一致；'
              f'变体间的**配对差值仍有效**，但绝对指标不可与该 arm 历史数字对读')
        return d
    k = min(5000, len(tg))
    same = bool(np.allclose(np.asarray(tg[:k], dtype=np.float64), targets_full[:k], rtol=1e-9, atol=0.0))
    print(f'[gate] {"✅" if same else "❌"} 前 {k} 行 targets 逐位比对{"相同" if same else "不同"} '
          f'→ 切分与行序{"可复现（绝对指标可与历史对读）" if same else "不可复现"}')
    return d


def make_recon(static_df, cache):
    """按 cid 复算 serve 侧可从网表得到的替代量：返回 (tc_recon, and_recon, inv_recon)。"""
    def _f(cid):
        if cid in cache:
            return cache[cid]
        try:
            nl = static_df.loc[cid, 'gate_level_netlist']
        except Exception:
            nl = ''
        cells = _cells_of(nl)
        tc = sum(float(gb.gate_struct(c)['n_t']) for c in cells)
        andr = sum(1 for c in cells if gb.gate_struct(c)['logic'] == 'AND')
        invr = sum(1 for c in cells if gb.gate_struct(c)['logic'] in ('INV', 'BUF'))
        cache[cid] = (tc, float(andr), float(invr))
        return cache[cid]
    return _f


# ---------------------------------------------------------------- 变体定义
def make_variants():
    """返回 [(name, desc, edit_x, csig_mode, prior_mode)]。
    edit_x=True → 输入引脚节点的 load 列改成 serve 的常数口径。
    csig_mode: true / serve(硬编码0) / zero1(只把 [1] 置 0) / recon(网表复算晶体管数)
    prior_mode: true / none / recon(网表复算的门类计数)"""
    return [
        ('base',              '训练口径：真 load + 真 csig + 真 prior',                 False, 'true',  'true'),
        ('serve',             'serve 现状：load=1e-15 + csig[1]=0 + prior=None',        True,  'serve', 'none'),
        ('csig0',             '只 csig[1]=0',                                           False, 'zero1', 'true'),
        ('csig0_fixTC',       'csig[1]=0 → 真晶体管数（oracle，量 csig 单项代价）',      False, 'true',  'true'),
        ('noprior',           '只 prior=None',                                          False, 'true',  'none'),
        ('loadconst',         '只 load=1e-15 常数',                                     True,  'true',  'true'),
        ('serve_fixTC',       'serve + csig[1]=真值（oracle）',                         True,  'true',  'none'),
        ('serve_fixTC_recon', 'serve + csig[1]=网表复算值（**可部署**）',               True,  'recon', 'none'),
        ('serve_fix_recon',   'serve + csig[1] 与 prior 都用网表复算（**可部署上界**）', True,  'recon', 'recon'),
        ('serve_fixTC_prior', 'serve + csig[1] 真值 + prior 真值（oracle 上界）',       True,  'true',  'true'),
        ('serve_noload',      'serve 但 load 用训练真值（分离 load 的边际贡献）',       False, 'serve', 'none'),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True, help='checkpoint(s)，多给则等权集成')
    ap.add_argument('--scaler', default=os.path.join('outputs', 'scaler.pkl'))
    ap.add_argument('--batches', default=os.environ.get('DATA_BATCHES',
                                                        'batch_v2_full,batch_v2_rest,batch_v2_m4'))
    ap.add_argument('--split-seed', type=int, default=int(os.environ.get('SPLIT_SEED', '42')))
    ap.add_argument('--min-group-size', type=int, default=int(os.environ.get('MIN_GROUP_SIZE', '10')))
    ap.add_argument('--max-exprs', type=int, default=150,
                    help='测试 expr 组抽样上限（0=全量；抽样按排序后等距取，确定性）。'
                         '1 个 expr 组 = 1 个候选集；150 组时比例量 SE ~3.9pp，'
                         '但变体间是**配对**比较，同一批次下差值 SE 远小于此')
    ap.add_argument('--variants', default='', help='逗号分隔的变体名子集（默认全部）')
    ap.add_argument('--cache-dir', default='')
    ap.add_argument('--out-json', default='')
    ap.add_argument('--verify-npz', default='auto',
                    help='切分自证用的 npz 路径；auto = 取 ckpt 同目录的 test_predictions.npz；none = 跳过')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.join('cache', f'serve_abl_{config.STRUCT_MODE}')
    print(f'[abl] config: STRUCT_MODE={config.STRUCT_MODE} USE_V2={config.USE_V2} '
          f'wave={config.USE_TRANSISTOR_WAVE} noise={config.USE_SUPPLY_NOISE} '
          f'caps={config.USE_PARASITIC_CAPS} struct_prior={config.USE_STRUCT_PRIOR} '
          f'ids_approx={os.environ.get("USE_IDS_AVG_APPROX", "0")} '
          f'IDS_GNN_TABLE={"set" if config.IDS_GNN_TABLE else "empty"}')

    static_parquets, dynamic_parquets, dyn, test_ids = build_test_frame(
        args.batches.split(','), args.min_group_size, args.split_seed)
    rebuild_vocab(static_parquets)
    probe_prior_tc(static_parquets)

    # 结构性约束：V2_STRUCT_MODE 必须在 DelayDataset 构建前生效（train_sweep.py:473-477）
    if config.USE_V2 and getattr(config, 'V2_STRUCT_MODE', None):
        config.STRUCT_MODE = config.V2_STRUCT_MODE

    scaler = load_scaler(args.scaler) if os.path.exists(args.scaler) else None
    if scaler is None:
        raise SystemExit(f'[abl] scaler 不存在: {args.scaler}（必须用该 arm 训练时的 scaler）')
    print(f'[abl] scaler: {args.scaler}  mean={np.round(scaler.mean_, 20).tolist()} '
          f'scale={np.round(scaler.scale_, 20).tolist()}  (序: slew, load, out_load, arrival)')

    test_df = dyn[dyn['circuit_id'].isin(test_ids)].reset_index(drop=True)
    n_rows_full, n_expr_full = len(test_df), test_df['expr'].nunique()
    # ---- 免费闸门：用 arm 自己的 npz targets 验重建切分（在任何抽样之前，行序即"全量"行序）----
    npz_path = args.verify_npz
    if npz_path == 'auto':
        npz_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt[0])), 'test_predictions.npz')
    elif npz_path == 'none':
        npz_path = ''
    _npz = verify_npz(npz_path, test_df['DELAY'].to_numpy(dtype=np.float64))
    if args.max_exprs and n_expr_full > args.max_exprs:
        exprs = sorted(test_df['expr'].astype(str).unique())
        stride = len(exprs) / float(args.max_exprs)
        keep = {exprs[int(i * stride)] for i in range(args.max_exprs)}
        test_df = test_df[test_df['expr'].astype(str).isin(keep)].reset_index(drop=True)
    print(f'[abl] test: {len(test_df)}/{n_rows_full} 行, '
          f'{test_df["expr"].nunique()}/{n_expr_full} 组（抽样后）')

    ds = RecDelayDataset(static_parquets, dynamic_parquets, test_ids, scaler, cache_dir,
                         dynamic_df=test_df, prefiltered=True)
    ds.last_dyn_keys = set()
    n = len(ds)
    print(f'[abl] dataset 行数 = {n}')

    # ---- 加载模型（必须用 serve.load_models：in_dim/词表/STRUCT_MODE 与部署同源）----
    c0 = ds.dynamic_df.iloc[0]
    cid0 = c0['circuit_id']
    srow = ds.static_df.loc[cid0]
    ip0, op0 = ds._circuit_pins.get(str(cid0), ([], []))
    models, in_dim = S.load_models(args.ckpt, scaler, srow['gate_level_netlist'], list(ip0), list(op0))
    device = torch.device(args.device)
    for m in models:
        m.eval(); m.to(device)
    print(f'[abl] 载入 {len(models)} 个模型, in_dim={in_dim}, ids_mode={S._APPROX_MODE}')

    variants = make_variants()
    if args.variants:
        want = {v.strip() for v in args.variants.split(',') if v.strip()}
        variants = [v for v in variants if v[0] in want]
        missing = want - {v[0] for v in variants}
        if missing:
            raise SystemExit(f'[abl] 未知变体: {sorted(missing)}')
    print(f'[abl] 变体: {[v[0] for v in variants]}')

    preds = {v[0]: np.full(n, np.nan) for v in variants}
    true_csigs = np.zeros((n, 3))
    serve_csigs = np.zeros((n, 3))
    diag = {'n_nodes_ne_gates': 0, 'dyn_node_frac': [], 'n_gates_missing': 0}
    _ngate_cache = {}
    recon_f = make_recon(ds.static_df, {})
    n_rec_rows = 0
    rec_tc_err, rec_prior_eq = [], 0

    t0 = time.time()
    for i in range(n):
        d = ds[i]
        row = ds.dynamic_df.iloc[i]
        cid = row['circuit_id']
        node_names, node_static, edge_index = ds._get_static(cid)
        base = node_static.shape[1]
        N = len(node_names)
        mask = torch.tensor([nm in ds.last_dyn_keys for nm in node_names],
                            dtype=torch.bool, device=device)
        diag['dyn_node_frac'].append(float(mask.float().mean()))

        csig_true = d.circuit_sig.to(device)                       # [1,3] 训练口径
        csig_srv = torch.tensor([[float(N), 0.0, float(csig_true[0, 2].item())]],
                                dtype=torch.float, device=device)  # serve.py:281 口径
        tc_recon, and_recon, inv_recon = recon_f(cid)
        csig_rec = torch.tensor([[float(N), float(tc_recon), float(csig_true[0, 2].item())]],
                                dtype=torch.float, device=device)
        prior_true = getattr(d, 'struct_prior', None)
        prior_true = prior_true.to(device) if prior_true is not None else None
        prior_rec = torch.tensor([[float(tc_recon), float(and_recon), float(inv_recon)]],
                                 dtype=torch.float, device=device)
        corner = d.corner_cond.to(device)
        ei = d.edge_index.to(device)
        batch = torch.zeros(N, dtype=torch.long, device=device)
        n_rec_rows += 1
        if csig_true[0, 1].item() > 0:
            _e = abs(tc_recon - float(csig_true[0, 1].item())) / float(csig_true[0, 1].item())
            rec_tc_err.append(_e)
        if prior_true is not None:
            rec_prior_eq += int(bool(abs(prior_true[0, 1].item() - and_recon) < 0.5
                                     and abs(prior_true[0, 2].item() - inv_recon) < 0.5))

        true_csigs[i] = csig_true[0].cpu().numpy()
        serve_csigs[i] = csig_srv[0].cpu().numpy()
        if cid not in _ngate_cache:
            _ngate_cache[cid] = len([l for l in str(ds.static_df.loc[cid, 'gate_level_netlist']).split('\n')
                                     if l.strip().startswith('X_')])
        n_gates_netlist = _ngate_cache[cid]
        if N != n_gates_netlist:
            diag['n_nodes_ne_gates'] += 1
        if float(csig_true[0, 1].item()) <= 0.0:
            diag['n_gates_missing'] += 1

        x_ok = d.x.to(device)
        x_loadconst = None
        if any(v[2] for v in variants):
            x_loadconst = x_ok.clone()
            load_scaled = (LOAD_F - float(scaler.mean_[1])) / float(scaler.scale_[1])
            x_loadconst[mask, base + 3] = load_scaled       # 动态 7 维里 load 的偏移 = 3

        for name, _desc, edit_x, csig_mode, prior_mode in variants:
            x = x_loadconst if edit_x else x_ok
            if csig_mode == 'serve':
                csig = csig_srv
            elif csig_mode == 'zero1':
                csig = torch.cat([csig_true[:, :1], torch.zeros(1, 1, device=device), csig_true[:, 2:]],
                                 dim=1)
            elif csig_mode == 'recon':
                csig = csig_rec
            else:
                csig = csig_true
            if prior_mode == 'none':
                prior = None
            elif prior_mode == 'recon':
                prior = prior_rec
            else:
                prior = prior_true
            with torch.no_grad():
                acc = 0.0
                for m in models:
                    out, _ = m(x, ei, batch, corner, csig, prior)
                    acc += float((10 ** out.cpu()).clamp(1e-12, 1e-8).item())
                preds[name][i] = acc / len(models)

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f'  .. {i+1}/{n} 行  ({el:.0f}s, {el/(i+1)*1000:.0f} ms/行)')

    print(f'[abl] 前向完成: {n} 行 x {len(variants)} 变体, {time.time()-t0:.0f}s')
    print(f'[abl] 诊断: num_nodes != 网表 X_ 行数的电路 = {diag["n_nodes_ne_gates"]}/{n} 行; '
          f'真 transistor_count <= 0 的行 = {diag["n_gates_missing"]}')
    print(f'[abl] 真 csig[0] (网表门数) 分位: '
          f'{np.percentile(true_csigs[:,0],[5,50,95]).round(1).tolist()}  '
          f'serve csig[0] (num_nodes) 分位: '
          f'{np.percentile(serve_csigs[:,0],[5,50,95]).round(1).tolist()}')
    print(f'[abl] 真 csig[1] (transistor_count) 分位: '
          f'{np.percentile(true_csigs[:,1],[5,50,95]).round(1).tolist()}  '
          f'(serve 恒为 0.0)')
    print(f'[abl] 输入引脚节点占比（拿到动态特征的节点 / 全图节点）中位: '
          f'{np.median(diag["dyn_node_frac"]):.3f}')
    if rec_tc_err:
        print(f'[abl] 网表复算晶体管数 vs 真值: 中位相对误差 {np.median(rec_tc_err)*100:.1f}%  '
              f'误差<1% 的占比 {np.mean([e < 0.01 for e in rec_tc_err])*100:.1f}%  (n={n_rec_rows})')
    print(f'[abl] 网表复算 prior 两项都命中的电路占比: {rec_prior_eq/max(n_rec_rows,1)*100:.1f}%')

    targets = ds.dynamic_df['DELAY'].to_numpy(dtype=np.float64)
    tdyn = ds.dynamic_df
    for name, _d, _e, _c, _p in variants:
        n_bad = int((~np.isfinite(preds[name])).sum())
        if n_bad:
            print(f'[abl] WARN: 变体 {name} 有 {n_bad}/{n} 行预测非有限值（会被排到最后，指标偏悲观）')
    out = {}
    print('\n' + '=' * 118)
    print('口径: ranking_metrics(avg_delay=True) / preds=10**out 线性 / hi_spread = 组内真实延迟差 >10%')
    print('=' * 118)
    hdr = (f'{"变体":<20s}{"全局遗憾%":>9s}{"Sp":>7s}{"top1%":>7s}  |'
           f'{"hi遗憾%":>8s}{"hiSp":>7s}{"hitop1%":>8s}  |'
           f'{"R2严":>6s}{"R2宽":>6s}{"hiR2严":>7s}{"hiR2宽":>7s}{"R3严":>6s}{"R3宽":>6s}')
    print(hdr)
    print('-' * 118)
    for name, desc, _e, _c, _p in variants:
        rk = ranking_metrics(tdyn, preds[name], targets, avg_delay=config.USE_V2)
        hi = rk.get('hi_spread', {})
        r2 = rk['recall_at_k'][2]; r3 = rk['recall_at_k'][3]
        hr2 = hi.get('recall_at_k', {}).get(2, {}); hr3 = hi.get('recall_at_k', {}).get(3, {})
        out[name] = {'desc': desc, 'global': rk, 'hi': hi}
        print(f'{name:<20s}{rk["regret_pct"]:>9.2f}{rk["spearman"]:>7.3f}{rk["top1_acc"]*100:>7.1f}  |'
              f'{hi.get("regret_pct", float("nan")):>8.2f}{hi.get("spearman", float("nan")):>7.3f}'
              f'{hi.get("top1_acc", float("nan"))*100:>8.1f}  |'
              f'{r2["strict"]["hit_pct"]*100:>6.1f}{r2["lenient"]["hit_pct"]*100:>6.1f}'
              f'{hr2.get("strict",{}).get("hit_pct",float("nan"))*100:>7.1f}'
              f'{hr2.get("lenient",{}).get("hit_pct",float("nan"))*100:>7.1f}'
              f'{r3["strict"]["hit_pct"]*100:>6.1f}{r3["lenient"]["hit_pct"]*100:>6.1f}')
    print('-' * 118)
    print(f'组数 n_groups={out[variants[0][0]]["global"]["n_groups"]}  '
          f'hi 组数={out[variants[0][0]]["hi"].get("n")}  '
          f'(R2/R3 严格=真#1 进预测前K；宽松=预测前K ∩ 真前K 非空；仅统计 m>=K+1 的非平凡组)')

    # ---- 可选：base 变体的预测 vs arm 自己的 npz（只在全量跑时有意义；行序已验证才可比）----
    if _npz is not None and 'base' in preds and args.max_exprs == 0 \
            and len(_npz['preds']) == n:
        bp = preds['base']
        np_ = np.asarray(_npz['preds'], dtype=np.float64)
        rel = np.abs(bp - np_) / np.maximum(np.abs(np_), 1e-30)
        print(f'[gate] base 变体 vs npz preds: 中位相对差 {np.median(rel):.3e}  '
              f'p95 {np.percentile(rel, 95):.3e}  最大 {rel.max():.3e}  '
              f'→ {"✅ 复刻成立（<1e-6）" if np.median(rel) < 1e-6 else "⚠ 有差异，看是否 epoch/集成口径不同"}')
    elif _npz is not None and 'base' in preds:
        print('[gate] （抽样跑或行数不符 → 跳过 base vs npz 的预测比对；要验就加 --max-exprs 0）')

    # ---- 相对 base 的差值 ----
    if 'base' in out:
        print('\n' + '=' * 118)
        print('相对 base 的差值（正 = 变差；遗憾 / hi遗憾 单位为 pp）')
        print('=' * 118)
        b = out['base']
        for name, desc, _e, _c, _p in variants:
            if name == 'base':
                continue
            rk, hi = out[name]['global'], out[name]['hi']
            bhi = b['hi']
            d_reg = rk['regret_pct'] - b['global']['regret_pct']
            d_hreg = hi.get('regret_pct', float('nan')) - bhi.get('regret_pct', float('nan'))
            d_r2 = (rk['recall_at_k'][2]['strict']['hit_pct']
                    - b['global']['recall_at_k'][2]['strict']['hit_pct']) * 100
            d_hr3 = (hi.get('recall_at_k', {}).get(3, {}).get('strict', {}).get('hit_pct', float('nan'))
                     - bhi.get('recall_at_k', {}).get(3, {}).get('strict', {}).get('hit_pct', float('nan'))) * 100
            print(f'{name:<20s} d遗憾={d_reg:+7.2f}pp  d hi遗憾={d_hreg:+7.2f}pp  '
                  f'd R2严={d_r2:+6.2f}pp  d hiR3严={d_hr3:+6.2f}pp   # {desc}')

    if args.out_json:
        ser = {k: {'desc': v['desc'],
                   'regret_pct': v['global']['regret_pct'],
                   'spearman': v['global']['spearman'],
                   'top1_acc': v['global']['top1_acc'],
                   'recall_at_k': {str(K): v['global']['recall_at_k'][K] for K in v['global']['recall_at_k']},
                   'hi_regret_pct': v['hi'].get('regret_pct'),
                   'hi_spearman': v['hi'].get('spearman'),
                   'hi_recall_at_k': {str(K): v['hi']['recall_at_k'][K]
                                      for K in v['hi'].get('recall_at_k', {})}}
               for k, v in out.items()}
        meta = {'ckpt': args.ckpt, 'scaler': args.scaler, 'batches': args.batches,
                'split_seed': args.split_seed, 'n_rows': n, 'n_rows_full': n_rows_full,
                'max_exprs': args.max_exprs, 'struct_mode': config.STRUCT_MODE,
                'n_groups': out[variants[0][0]]['global']['n_groups']}
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump({'meta': meta, 'variants': ser}, f, ensure_ascii=False, indent=2)
        print(f'\n[abl] 已写 {args.out_json}')


if __name__ == '__main__':
    main()
