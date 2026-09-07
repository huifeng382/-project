"""服务器全量版: 复刻 DelayGNN → 独立 per-gate ids_avg 回归 (DATA_BATCHES = full+rest+m4 同 delayGNN 训练域)

与 _fit_idsavg_gnn.py 同架构/同口径 (电路级切分, 唯一尺度 = ids_avg R^2/Spearman), 差异:
  - 数据 = DATA_BATCHES (默认 batch_v2_full,batch_v2_rest,batch_v2_m4; 与 config.DATA_BATCHES 一致)
    * rest 批次为 *_partN.parquet 分片, 自动 glob
  - 行按电路预分组 (groupby), 避免 45k 电路逐行过滤
  - 测试按批次来源分桶 (full/rest/m4), 验证 m4 等未见形态的泛化

本地受控对照 (1500 电路, batch_v2_full): GBDT15=0.6740 / gnn=0.7697 / nograph=0.6773
  => 图传播携带 per-gate ids_avg 信号; 本脚本在全量数据 (含 m4) 上复核.

用法 (服务器, 训练在服务器跑 = 用户执行):
  DATA_BATCHES='batch_v2_full,batch_v2_rest,batch_v2_m4' python scripts/diag/_fit_idsavg_gnn_server.py
  可选: N_CAP=<电路数上限> 快速验证; EPOCHS=<n> 默认 45; NO_NOGRAPH=1 跳过无边对照(省~一半时间)
  17.0.5: K/HID/EMB/DROPOUT/LR/PATIENCE 可 env 覆盖(0=不早停); CONE_FEAT=1 追加锥体/距离通道(见 17.0.5 记录)
  17.0.8: CONE_V2=1 锥体 v2 (含 v1 3通道 + 主通路/锥内扇出/锥内扇入, N_EXTRA 10->13); 17.0.7 A2/B2 裁决见 docs
  17.0.11: SCHED=rlp 用 ReduceLROnPlateau(以 val R^2 平台降 LR) 解耦调度与 EPOCHS 上限; 默认 cosine 逐位等价; 17.0.9 Arrow 切分修复/17.0.10 长程裁决见 docs
  17.0.14: 默认升为 B2v2 好基座 (K5/H160/EMB32/LR1e-3/CONE_V2=1/NO_NOGRAPH=1) + 更快早停收口
           (EPOCHS=400 仅安全上限; LR_TMAX=120 = cosine 退火期; PATIENCE=40/STOP_EPS=2e-3 = 阈值敏感早停)
           ⚠ 裸跑不再=17.0.1; 复现任何历史配置须显式 env 传旧值 (见 docs/IDS_AVG_GNN_R2_DIRECTIONS.md)
  17.0.16: ②' 激活锥/腿数代理 CONE_PIN=1 (输入脚级锥粒度: 每门 driver∈d1 的输入计数 count/frac/ge2,
           恒放 extras 尾部, N_EXTRA+3; 须 cone d1 可达, 与 CONE_V2/CONE_FEAT 正交叠加或单开);
           修 17.0.15 dump 守卫 bug (N_EXTRA>=7 → 语义守卫 CONE_N) + log 头配置回显 [CFG].
           R4 = 17.0.14 base (CONE_V2=1) + CONE_PIN=1, 与 R1 base 锚 (0.7861) 同 regime 对照
  17.0.17: n_t 条件专家旋钮 TIER_ONLY (''/ge8/le2/3-7; 空串=行为不变)。判定口径:
           默认 17.0.14 base (CONE_V2=1 无 CONE_PIN) + TIER_ONLY=ge8 跑专化模型, 其 test R² = n>=8 层 R²,
           与 R1 base 锚 n>=8 层 0.5689 (同切分同 regime, 152k test 行) 直接可比; 显著 > 0.5689 → 复合路由有价值.
"""
import sys, os, json, math, time, glob as _glob
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
from src.graph_builder import build_static_graph, rebuild_gate_types

torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEV={DEV}', flush=True)

DATA_ROOT = os.environ.get('DATA_ROOT', 'data')
DATA_BATCHES = os.environ.get('DATA_BATCHES', 'batch_v2_full,batch_v2_rest,batch_v2_m4')
N_CAP = int(os.environ.get('N_CAP', '0') or 0)     # 0=全部电路
EPOCHS = int(os.environ.get('EPOCHS', '400'))           # 17.0.14 起 = 安全上限 (早停/阈值收口决定实际跑长); 须 > LR_TMAX
NO_NOGRAPH = os.environ.get('NO_NOGRAPH', '1') == '1'    # 17.0.14 默认跳无边对照 W (本地已证 gnn-nograph=+0.09); 复现旧档设 0
CIRC_SPLIT = (0.85, 0.05, 0.10)
# GNN 超参默认 = B2v2 好基座 (17.0.6 A2 容量 + 17.0.8 cone v2 裁决); 17.0.5 起全部可 env 覆盖
EMB = int(os.environ.get('EMB', '32')); HID = int(os.environ.get('HID', '160'))
K = int(os.environ.get('K', '5')); DROPOUT = float(os.environ.get('DROPOUT', '0.25'))
LR = float(os.environ.get('LR', '1e-3'))
PATIENCE = int(os.environ.get('PATIENCE', '40') or 0)   # 距最后"有意义进步"40ep 收 (eval 每5ep); 0=不早停
CONE_FEAT = os.environ.get('CONE_FEAT', '0') == '1'      # 锥体 v1 通道 (需显式开; 默认走 V2)
CONE_V2   = os.environ.get('CONE_V2', '1') == '1'        # 17.0.8 锥体 v2; 17.0.14 默认开 (=B2v2 基座); 复现无锥体设 0
# ②' 17.0.16 候选 (R4): 激活锥/腿数代理 = 输入脚级锥粒度。锥体粗量(门在不在切换锥上)已贡献 +0.045;
#   本旋钮给每门标"这次切换锥拉动了它几条输入脚"(driver∈d1 的输入计数) —— serve 可行的 log(n_on) 代理
#   (标签向量间运动 84% 由 log(n_on) 驱动(§0.5); 残差集中 n>=8 堆叠门(§0.6 ③))。三通道恒放 extras 尾部:
#   count / frac(=count/输入脚总数) / ge2(1=多腿同拉)。须 d1 可达; 与 CONE_V2/CONE_FEAT 正交叠加或单开。
CONE_PIN = os.environ.get('CONE_PIN', '0') == '1'        # ②' 输入脚级激活锥通道 (默认关; R4 = 17.0.14 base + 本旋钮=1)
# ① 17.0.15 候选: STRUCT_MODE 注入 graph_builder ('rich' = base 7 通道 + stack/parallel 两列, serve 零成本);
#   ns 列序: col0=type, col1..6=fan/depth/drive/parasitic/logic_effort/h, col7=n_t, col8/9=stack/parallel(rich)
#   rich 与 base 前 7 连续通道同公式同序 => 逐位一致, 无静默改口径 (17.0.15 本地已验)
STRUCT_MODE = os.environ.get('STRUCT_MODE', 'base')
import config as _cfg
_cfg.STRUCT_MODE = STRUCT_MODE
N_CONT_BASE = int(os.environ.get('N_CONT_BASE', '9' if STRUCT_MODE == 'rich' else '7'))   # base=7 (止于 n_t) / rich=9 (+stack/parallel)
CONE_N = 6 if CONE_V2 else (3 if CONE_FEAT else 0)       # 锥体连续通道数: v2=6 / v1=3 / 无=0 (cones 从 extras +7 起, 见 assemble)
PIN_N  = 3 if CONE_PIN else 0                            # ②' 输入脚通道数 (恒在 extras 尾部)
N_EXTRA = 7 + CONE_N + PIN_N                             # base 7 extras(+0..+6) + 锥体块 + ②' 尾部块
# 17.0.17 候选 (并行 R5): n_t 条件专家 —— 只在该层上训练/评估 (镜像 m4 特化经验: 让模型专化难层)。
#   判定口径: 默认 17.0.14 base (CONE_V2=1 无 CONE_PIN) + TIER_ONLY=ge8 跑一个专化模型, 其 test R² 即 n>=8 层 R²,
#   与 R1 base 锚的 n>=8 层 0.5689 (同切分同 regime, 152k test 行) 直接可比; 显著 > 0.5689 => 复合(按 n_t 路由)才有意义。
#   sup 过滤点: 采样门循环按 ns[i,7]=n_t (GBDT 与 GNN 同吃子集); 无命中门的整行(向量)照旧跳过。
TIER_ONLY = os.environ.get('TIER_ONLY', '')               # ''=全层(默认, 行为不变) | 'ge8'=n_t>=8 | 'le2'=n_t<=2 | '3-7'=3<=n_t<=7
def _tier_ok(nt):
    if not TIER_ONLY: return True
    if TIER_ONLY == 'ge8': return nt >= 8
    if TIER_ONLY == 'le2': return nt <= 2
    if TIER_ONLY == '3-7': return 3 <= nt <= 7
    raise ValueError(f'TIER_ONLY={TIER_ONLY!r} 未知 (空/ge8/le2/3-7)')
# 17.0.11: SCHED 解耦调度 —— 'cosine'(默认, T_max=EPOCHS 逐位不变) | 'rlp'(ReduceLROnPlateau: val R^2 平台降 LR, 与 EPOCHS 上限无关)
SCHED = os.environ.get('SCHED', 'cosine')
RLP_FACTOR = float(os.environ.get('RLP_FACTOR', '0.5'))
RLP_PAT = int(os.environ.get('RLP_PAT', '8'))          # 单位=eval 点(每 5ep 一 eval); 连续 RLP_PAT 个 eval 无 >=RLP_THRESH 改善 → LR×factor
RLP_THRESH = float(os.environ.get('RLP_THRESH', '2e-3'))  # abs(原始 R^2): val 需超当前 best ≥ 此值才算改善
RLP_COOL = int(os.environ.get('RLP_COOL', '2'))        # LR 降后冷却(eval 点数)
RLP_MINLR = float(os.environ.get('RLP_MINLR', '1e-6'))
# 17.0.14: 早停收口 (上不封顶, EPOCHS 只当安全上限) —— LR 预算期 + 阈值敏感早停 (默认已启用; 复现历史设 LR_TMAX=0/STOP_EPS=0/PATIENCE=0)
LR_TMAX  = int(os.environ.get('LR_TMAX', '120') or 0)    # cosine 退火期=LR_TMAX (到期冻结 LR, 不与 EPOCHS 绑死); 0=历史 (T_max=EPOCHS)
STOP_EPS = float(os.environ.get('STOP_EPS', '2e-3') or 0) # 仅当 val 提升 ≥STOP_EPS 视为"有意义进步"并重置早停钟; 0=历史 (任何抬升都重置)
# ③ 17.0.15 候选: ckpt 落盘 (best_sd + 模型元信息) + test 残差/属性 dump (供本地分析 残差 vs n_t/锥深, 判 ②' 激活锥值不值投)
_sck = os.environ.get('SAVE_CKPT', '')
CKPT_PATH = ('idsavg_gnn_best.pt' if _sck == '1' else _sck)   # 非空即落盘
DUMP_PATH = os.environ.get('DUMP_RESID', '')                   # 非空即 dump test 每 (row,门) 预测/真值/属性(n_t/锥深/type/batch)
# 17.0.16: log 头配置回显 —— 防 R3 式静默误配 (跑前先核这行配置对不对; 候选必须与想跑的完全一致)
print('[CFG] ' + ' | '.join([
    f'DATA={DATA_BATCHES}', f'N_CAP={N_CAP}',
    f'STRUCT_MODE={STRUCT_MODE}', f'N_CONT_BASE={N_CONT_BASE}',
    f'CONE_V2={int(CONE_V2)}', f'CONE_FEAT={int(CONE_FEAT)}', f'CONE_PIN={int(CONE_PIN)}',
    f'CONE_N={CONE_N}', f'PIN_N={PIN_N}', f'N_EXTRA={N_EXTRA}', f'n_cont={N_CONT_BASE + N_EXTRA}',
    f'K={K}', f'HID={HID}', f'EMB={EMB}', f'LR={LR}', f'DROPOUT={DROPOUT}',
    f'EPOCHS={EPOCHS}', f'LR_TMAX={LR_TMAX}', f'STOP_EPS={STOP_EPS}', f'PATIENCE={PATIENCE}', f'SCHED={SCHED}',
    f'NO_NOGRAPH={int(NO_NOGRAPH)}', f'CKPT={CKPT_PATH or "-"}', f'DUMP={DUMP_PATH or "-"}',
    f'TIER_ONLY={TIER_ONLY or "-"}',
]), flush=True)

def parse_corner(corner):
    try:
        s, l = str(corner).split('_')[:2]
        return float(s[1:].replace('p', '.')), float(l[1:].replace('p', '.'))
    except Exception:
        return 5.0, 10.0

def cell_types(nl):
    return {ln.split()[-1] for ln in (nl or '').split('\n')
            if ln.strip().startswith('X_') and len(ln.split()) >= 3}

def resolve_parquets(batch):
    """返回 (static_paths, dynamic_paths), 支持 *_partN.parquet 分片 (对齐 train_sweep 逻辑)."""
    sp = os.path.join(DATA_ROOT, f'{batch}/circuit_static.parquet')
    dp = os.path.join(DATA_ROOT, f'{batch}/timing_arcs.parquet')
    if os.path.exists(sp) and os.path.exists(dp):
        return [sp], [dp]
    sparts = sorted(_glob.glob(os.path.join(DATA_ROOT, f'{batch}/circuit_static_part*.parquet')))
    dparts = sorted(_glob.glob(os.path.join(DATA_ROOT, f'{batch}/timing_arcs_part*.parquet')))
    if sparts and dparts:
        return sparts, dparts
    raise FileNotFoundError(f'batch {batch}: no static/dynamic parquet found under {DATA_ROOT}')

# ============================================================ 载入多批次
_batch_names = [b.strip() for b in DATA_BATCHES.split(',') if b.strip()]
static_dfs, dyn_dfs = [], []
batch_of_circ = {}     # circuit_id -> 来源批次 (第一命中为准; 与 drop_duplicates 序一致)
for batch in _batch_names:
    try:
        sps, dps = resolve_parquets(batch)
    except FileNotFoundError as e:
        print(f'WARN: {e}, skip'); continue
    s = pd.read_parquet(sps[0] if len(sps) == 1 else sps) if len(sps) == 1 else pd.concat([pd.read_parquet(p) for p in sps])
    d = pd.concat([pd.read_parquet(p, columns=['circuit_id', 'transistor_wave_json', 'slew_s',
                                               'output_load_f', 'corner', 'direction',
                                               'switching_pin', 'output']) for p in dps], ignore_index=True)
    static_dfs.append(s); dyn_dfs.append(d)
    for c in s['circuit_id'].astype(str):
        batch_of_circ.setdefault(c, batch)
    print(f'batch {batch}: static={len(s)} circuits / dynamic={len(d)} rows', flush=True)

sdf = pd.concat(static_dfs).drop_duplicates('circuit_id')
sdf['circuit_id'] = sdf['circuit_id'].astype(str)
sdf = sdf.set_index('circuit_id')
ddf = pd.concat(dyn_dfs, ignore_index=True)
ddf['circuit_id'] = ddf['circuit_id'].astype(str)
ddf = ddf[ddf['transistor_wave_json'].notna()]
print(f'总: 电路 {len(sdf)} / 行 {len(ddf)}', flush=True)

# 电路级切分
# ⚠ 17.0.9 修复: circuit_id 为 pyarrow-backed string, set()/list() 逐元素走 arrow.__iter__
#   → 746k 行实测 30min+ 仍卡 (py-spy 定位; 此前 A2/B2 也在此磨 ~30-45min 未被察觉, N_CAP 探针"停滞"同源)
#   先 C 速 to_numpy() 转 object 再做 set (语义相同, 亚秒级)。
_cid_set = set(ddf['circuit_id'].to_numpy())
circ_all = [c for c in sdf.index if c in _cid_set]
if N_CAP > 0:
    circ_all = np.random.RandomState(42).choice(circ_all, size=min(N_CAP, len(circ_all)), replace=False).tolist()
circ_all = list(circ_all)
order = np.asarray(circ_all); rp = np.random.RandomState(7).permutation(len(order))
n1 = int(len(order)*CIRC_SPLIT[0]); n2 = n1 + int(len(order)*CIRC_SPLIT[1])
tr_c = set(order[rp[:n1]].tolist()); va_c = set(order[rp[n1:n2]].tolist()); te_c = set(order[rp[n2:]].tolist())
# 有序列表 (set 迭代序不确定, assemble/分桶一律用有序列表保证块顺序可复现)
tr_l = [c for c in circ_all if c in tr_c]
va_l = [c for c in circ_all if c in va_c]
te_l = [c for c in circ_all if c in te_c]
print(f'电路级: train {len(tr_l)} / val {len(va_l)} / test {len(te_l)}', flush=True)

# 行预分组 (只保留需要用到的电路)
use = tr_c | va_c | te_c
ddf = ddf[ddf['circuit_id'].isin(use)]
rowg = {cid: g for cid, g in ddf.groupby('circuit_id')}
del ddf

# ============================================================ 逐电路建图 + 收集样本/块
Xs, ys, row_m = [], [], []
blocks = {}; max_type = 0

t0 = time.time()
for ci, cid in enumerate(circ_all):
    srow = sdf.loc[cid]
    nl = srow['gate_level_netlist']
    try:
        ip = json.loads(srow['input_pins_json']) if isinstance(srow['input_pins_json'], str) else srow['input_pins_json']
        op = json.loads(srow['output_pins_json']) if isinstance(srow['output_pins_json'], str) else srow['output_pins_json']
        rebuild_gate_types(cell_types(nl))
        node_names, ns, ei = build_static_graph(cid, nl, ip or None, op or None)
    except Exception:
        continue
    ns = ns.numpy()
    max_type = max(max_type, int(ns[:, 0].max()))
    n2i = {n: i for i, n in enumerate(node_names)}
    edges = [(int(a), int(b)) for a, b in ei.t().tolist() if a != b]
    rows_df = rowg.get(cid)
    if rows_df is None:
        continue
    blk = []
    for _, r in rows_df.iterrows():
        try:
            wave = json.loads(r['transistor_wave_json']) if isinstance(r['transistor_wave_json'], str) else {}
        except Exception:
            continue
        if not isinstance(wave, dict) or not wave:
            continue
        gate_avg = {}
        for tv in wave.values():
            if not isinstance(tv, dict): continue
            g, v = tv.get('gate'), tv.get('ids_avg')
            if g is None or v is None: continue
            gate_avg.setdefault(str(g).lower(), []).append(float(v))
        if not gate_avg: continue
        slew_s = float(r.get('slew_s', 0) or 0); load_f = float(r.get('output_load_f', 0) or 0)
        c_slew, c_load = parse_corner(r.get('corner'))
        row_slew = (slew_s if slew_s > 0 else c_slew*1e-12)
        row_load = (load_f if load_f > 0 else c_load*1e-15)
        f_slew = math.log1p(row_slew*1e12); f_load = math.log1p(row_load*1e15)
        f_cs = math.log1p(c_slew); f_cl = math.log1p(c_load)
        dir_code = 1.0 if str(r.get('direction')) == 'rise' else 0.0
        src_i = n2i.get(str(r.get('switching_pin', '')).lower())
        out_i = n2i.get(str(r.get('output', '')).lower())
        sup = []
        for i, n in enumerate(node_names):
            gk = str(n).lower()
            if gk not in gate_avg: continue
            real = float(np.mean(gate_avg[gk]))
            if real <= 0: continue
            if not _tier_ok(float(ns[i, 7])): continue   # 17.0.17 n_t 条件专家: 只留该层采样门 (GBDT+GNN 同步收窄)
            y = math.log1p(real)
            sup.append((i, y))
            drive, par = float(ns[i, 3]), float(ns[i, 4])
            fan, hh = float(np.expm1(ns[i, 1])), float(np.expm1(ns[i, 6]))
            dep, g = float(ns[i, 2]), float(ns[i, 5])
            f_d, f_p = math.log1p(drive), math.log1p(par)
            f_fan, f_h = math.log1p(fan), math.log1p(hh)
            r_on = 1.0/max(drive, 1e-6); c_l = max(par*1e-15, 1e-18)
            Xs.append([f_slew, f_load, f_d, f_p, f_fan, f_h, f_cs, f_cl,
                       f_d*f_load, f_d*f_h, f_fan*f_p, f_slew*f_load,
                       math.log1p(r_on*c_l*1e15), math.log1p(max(par*fan, 1e-9)),
                       math.log1p(max(1.0/(1.0+fan), 1e-9)), dep, g])
            ys.append(y); row_m.append(cid)
        if not sup: continue
        blk.append({'edges': edges, 'f_slew': f_slew, 'f_load': f_load, 'f_cs': f_cs,
                    'f_cl': f_cl, 'dir_code': dir_code, 'src_i': src_i, 'out_i': out_i, 'sup': sup})
    if blk:
        _be = {'ns': ns, 'rows': blk, 'batch': batch_of_circ.get(cid, '?')}
        if (CONE_FEAT or CONE_V2 or CONE_PIN) and edges:      # B/V2/②': 每电路邻接表一次建好, assemble 三份 (tr/va/te) 复用
            _adj = {}; _radj = {}
            for _a, _b in edges:
                _adj.setdefault(_a, []).append(_b); _radj.setdefault(_b, []).append(_a)
            _be['adj'] = _adj; _be['radj'] = _radj
        blocks[cid] = _be
    if (ci+1) % 5000 == 0:
        print(f'  电路 {ci+1}/{len(circ_all)}: GBDT样本 {len(ys)} / 块 {len(blocks)}, {time.time()-t0:.0f}s', flush=True)

Xs = np.array(Xs); ys = np.array(ys)
NUM_TYPES = max_type + 2
print(f'\n样本 {len(ys)} (row,gate); 电路块 {len(blocks)}; max_type={max_type}', flush=True)

_tr_ns = np.vstack([blocks[c]['ns'][:, 1:1+N_CONT_BASE] for c in tr_c if c in blocks])
_sta_mean = _tr_ns.mean(0); _sta_std = _tr_ns.std(0) + 1e-6

def masks(cids):
    m = np.array([c in cids for c in row_m])
    return m
mtr = masks(tr_c); mva = masks(va_c); mte = masks(te_c)
print('GBDT 样本量 train/val/test:', mtr.sum(), mva.sum(), mte.sum(), flush=True)

# ============================================================ A. GBDT15 基线
print('\n===== A. GBDT15 (15特征, 部署同款, 电路级切分) =====', flush=True)
def gbdt_r2(name, Xtr, ytr, Xte, yte):
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=42,
                                       validation_fraction=0.1, early_stopping=True, n_iter_no_change=40)
    gb.fit(Xtr, ytr)
    p = gb.predict(Xte)
    r2 = 1-np.sum((yte-p)**2)/np.sum((yte-yte.mean())**2)
    rho, _ = spearmanr(yte, p)
    print(f'  {name:34s} R^2={r2:.4f}  Spearman={rho:.4f}  n_iter={gb.n_iter_}', flush=True)
    return gb, p
gb, pte = gbdt_r2('A. GBDT15', Xs[mtr][:, :15], ys[mtr], Xs[mte][:, :15], ys[mte])

# test 每样本 GBDT 预测按电路聚合 (分桶复用; 避免逐电路 O(N) 扫描)
_carr = np.array(row_m)
gb_by_cid = {}
for j, pos in enumerate(np.where(mte)[0]):
    gb_by_cid.setdefault(_carr[pos], []).append(pte[j])

# ============================================================ DelayGNN 复刻骨架
class GraphConvL(nn.Module):
    def __init__(self, fin, fout, bias=True):
        super().__init__()
        self.W1 = nn.Linear(fin, fout, bias=bias)
        self.W2 = nn.Linear(fin, fout, bias=False)
    def forward(self, x, edge_index, n_nodes):
        out = self.W1(x)
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros(n_nodes, self.W2.out_features, device=x.device, dtype=x.dtype)
            agg.index_add_(0, dst, self.W2(x[src]))
            out = out + agg
        return out

class IdsAvgGNN(nn.Module):
    def __init__(self, num_types, n_cont, emb=EMB, hid=HID, K=K, dropout=DROPOUT):
        super().__init__()
        self.gate_embed = nn.Embedding(num_types, emb)
        actual_in = emb + n_cont
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        self.convs.append(GraphConvL(actual_in, hid)); self.norms.append(nn.LayerNorm(hid))
        for _ in range(K - 1):
            self.convs.append(GraphConvL(hid, hid)); self.norms.append(nn.LayerNorm(hid))
        self.head = nn.Linear(hid, 1)
        self.dropout = dropout
    def forward(self, type_idx, cont, edge_index, n_nodes):
        x = torch.cat([self.gate_embed(type_idx), cont], dim=-1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index, n_nodes)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0 and residual.shape == x.shape:
                x = x + residual
        return self.head(x).squeeze(-1)

def _cone_dists(start, adj):
    """BFS 沿有向边 (driver->receiver) 从 start 出发: 返回 {节点: 跳数}. 无边时仅含 start."""
    dist = {start: 0}; dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1; dq.append(v)
    return dist

def assemble(cids, use_edges=True):
    out = []
    for cid in cids:
        b = blocks.get(cid)
        if not b: continue
        ns = b['ns']; N = ns.shape[0]; rows = b['rows']; R = len(rows)
        static = (ns[:, 1:1+N_CONT_BASE].astype(np.float32) - _sta_mean) / _sta_std
        typ = ns[:, 0].astype(np.int64)
        T = np.zeros((R*N, N_CONT_BASE + N_EXTRA), dtype=np.float32)
        Ty = np.empty(R*N, dtype=np.int64)
        for rk, row in enumerate(rows):
            s = rk*N; e = s+N
            T[s:e, :N_CONT_BASE] = static
            T[s:e, N_CONT_BASE+0] = row['f_slew']
            T[s:e, N_CONT_BASE+1] = row['f_load']
            T[s:e, N_CONT_BASE+2] = row['f_cs']
            T[s:e, N_CONT_BASE+3] = row['f_cl']
            T[s:e, N_CONT_BASE+4] = row['dir_code']
            if row['src_i'] is not None: T[s+row['src_i'], N_CONT_BASE+5] = row['f_slew']
            if row['out_i'] is not None: T[s+row['out_i'], N_CONT_BASE+6] = row['f_load']
            if CONE_N or CONE_PIN:
                # B/V2/②': 锥体块(宽 CONE_N) 恒从 extras +7 起, 原锚位 +5/+6 不变; ②' 三通道恒在尾部, 总宽 Wc=CONE_N+PIN_N
                #   cone 块: +7 在 src 扇出锥内(含 src) | +8 沿有向边下游深度 (src=0, 最深=1)
                #           +9 反向(朝 out)上游深度; v2 追加 +10 主通路 / +11 锥内扇出 / +12 锥内扇入
                #   src_i/out_i=None => 相应通道全 0; d1 在 src_i 存在时恒算 (②' 也要用)
                Wc = CONE_N + PIN_N
                co = np.zeros((N, Wc), dtype=np.float32)
                d1 = {}
                if row['src_i'] is not None:
                    d1 = _cone_dists(row['src_i'], b.get('adj', {}))
                if CONE_N:
                    if d1:
                        md1 = max(d1.values())
                        for g, dd in d1.items():
                            co[g, 0] = 1.0; co[g, 1] = dd / max(md1, 1)
                    if row['out_i'] is not None:
                        d2 = _cone_dists(row['out_i'], b.get('radj', {}))
                        if d2:
                            md2 = max(d2.values())
                            for g, dd in d2.items():
                                co[g, 2] = dd / max(md2, 1)
                if CONE_V2:
                    # V2 +3 通道 (17.0.8), 复用 d1/d2 同一次 BFS —— 描述每扇门在开关事件里的"角色":
                    #   +3 主通路 (∈d1∧∈d2, 0/1) = 须驱动输出负载的贯穿门; +4 锥内扇出(child∈d1)=分支/叶
                    #   +5 锥内扇入(parent∈d1)=再汇聚 (多路信号叠加 → 电流大); d1/d2 为空 => 该通道 0
                    if row['src_i'] is not None and row['out_i'] is not None and d1 and d2:
                        for g in d1:
                            if g in d2: co[g, 3] = 1.0
                    if d1:
                        _adj = b.get('adj', {}); _radj = b.get('radj', {})
                        _mxo = _mxi = 0
                        for g in d1:
                            _fo = sum(1 for c in _adj.get(g, ()) if c in d1); co[g, 4] = _fo
                            _fi = sum(1 for p in _radj.get(g, ()) if p in d1); co[g, 5] = _fi
                            if _fo > _mxo: _mxo = _fo
                            if _fi > _mxi: _mxi = _fi
                        if _mxo > 0:
                            for g in d1: co[g, 4] /= _mxo
                        if _mxi > 0:
                            for g in d1: co[g, 5] /= _mxi
                if CONE_PIN and d1:
                    # ②' 输入脚级激活锥 (17.0.16 R4): 该门"几条输入脚被这次切换锥拉动" —— g 的 driver 中∈d1 的计数
                    #   count: 输入条数(其输出随动 → 该输入对应腿被拉); frac: count/该门输入脚总数 (归一腿占比);
                    #   ge2: 1 若 count>=2 (多腿同拉 → 深堆叠/并联门电流最敏感区, = §0.6 n>=8 残差池)
                    #   count>0 ⟺ g∈d1 锥内 (driver→g 可达); 种子 src 自身无 in-d1 driver → 恒 0 (区分源 vs 锥内)
                    _radj = b.get('radj', {})
                    for g in d1:
                        _drv = _radj.get(g, ()); _nt = len(_drv)
                        if not _nt: continue
                        _cnt = sum(1 for _p in _drv if _p in d1)
                        co[g, CONE_N + 0] = float(_cnt)
                        co[g, CONE_N + 1] = _cnt / _nt
                        co[g, CONE_N + 2] = 1.0 if _cnt >= 2 else 0.0
                T[s:e, N_CONT_BASE+7 : N_CONT_BASE+7+Wc] = co
            Ty[s:e] = typ
        edges_t = []
        if use_edges:
            for rk in range(R):
                off = rk*N
                edges_t += [(a+off, b+off) for a, b in rows[0]['edges']]
        ei = torch.tensor(edges_t, dtype=torch.long).t().contiguous() if edges_t else torch.zeros(2, 0, dtype=torch.long)
        sup_i = [rk*N + gi for rk, row in enumerate(rows) for gi, _ in row['sup']]
        sup_y = [y for row in rows for _, y in row['sup']]
        out.append((torch.tensor(Ty), torch.tensor(T), ei,
                    torch.tensor(sup_i, dtype=torch.long), torch.tensor(sup_y, dtype=torch.float32)))
    return out

n_cont = N_CONT_BASE + N_EXTRA
GSZ = int(os.environ.get('GSZ', '1'))      # ⚠ 默认 1 = 逐电路步进(已验证稳定)。GSZ>1 拼接提速为实验性(见 run_full_local 记录),暂勿开
def concat_group(group):
    tys, cos, eis, sis, sys_, n_off, has_e = [], [], [], [], [], 0, False
    for ty, co, ei, s_i, s_y in group:
        N = ty.shape[0]
        tys.append(ty); cos.append(co)
        if ei.numel():
            has_e = True; eis.append(ei + n_off)
        sis.append(s_i + n_off); sys_.append(s_y)
        n_off += N
    Ty = torch.cat(tys); Co = torch.cat(cos); Si = torch.cat(sis); Sy = torch.cat(sys_)
    E = torch.cat(eis, dim=1) if has_e else torch.zeros(2, 0, dtype=torch.long)
    return Ty, Co, E, Si, Sy

def run_variant(name, tr_data, va_data, te_data, out_ckpt=None):
    model = IdsAvgGNN(NUM_TYPES, n_cont).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if SCHED == 'rlp':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=RLP_FACTOR, patience=RLP_PAT,
            threshold=RLP_THRESH, threshold_mode='abs', cooldown=RLP_COOL, min_lr=RLP_MINLR)
    else:
        # 17.0.14: cosine 退火期可独立于 EPOCHS (LR_TMAX>0); 到期冻结 LR, 交给阈值早停收口
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(LR_TMAX if LR_TMAX > 0 else EPOCHS))
    best_va = -1e9; best_sd = None; best_ep = 0; last_imp_ep = 0
    def eval_blocks(datas):
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for ty, co, ei, s_i, s_y in datas:
                ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV); s_i = s_i.to(DEV)
                ps.append(model(ty, co, ei, ty.shape[0])[s_i].cpu().numpy()); ts.append(s_y.numpy())
        if not ps: return -9, -9, 0
        ps = np.concatenate(ps); ts = np.concatenate(ts)
        r2 = 1-np.sum((ts-ps)**2)/np.sum((ts-ts.mean())**2)
        rho, _ = spearmanr(ts, ps)
        if os.environ.get('DEBUG') == '1':
            print(f'      [dbg] pred mean={ps.mean():.4f} std={ps.std():.4f} min={ps.min():.4f} max={ps.max():.4f} | y mean={ts.mean():.4f} std={ts.std():.4f} min={ts.min():.4f} max={ts.max():.4f}', flush=True)
        return r2, rho, len(ts)
    for ep in range(EPOCHS):
        model.train()
        el, nb = 0.0, 0
        for i in range(0, len(tr_data), GSZ):
            ty, co, ei, s_i, s_y = concat_group(tr_data[i:i+GSZ])
            ty = ty.to(DEV); co = co.to(DEV); ei = ei.to(DEV)
            s_i = s_i.to(DEV); s_y = s_y.to(DEV)
            pred = model(ty, co, ei, ty.shape[0])[s_i]
            loss = F.mse_loss(pred, s_y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += s_y.shape[0]
        if SCHED == 'cosine' and (LR_TMAX == 0 or ep < LR_TMAX):
            sched.step()   # cosine 逐 ep 退火 (历史逐位一致; 17.0.14 LR_TMAX>0 时退火期=LR_TMAX, 到期冻结不再退火重启); rlp 在下方 eval 后喂 val 步进
        if (ep+1) % 5 == 0 or ep == 0:
            vr2, vrho, n = eval_blocks(va_data)
            lr_now = opt.param_groups[0]['lr']
            if SCHED == 'rlp':
                sched.step(vr2)   # val R^2 平台降 LR (thr/patience 见 17.0.11)
                lr_now = opt.param_groups[0]['lr']
            print(f'  [{name}] ep {ep+1:2d}  lr={lr_now:.1e}  train_loss={el/max(nb,1):.5f}  val_R^2={vr2:.4f} (n={n})', flush=True)
            prev_va = best_va
            if vr2 > best_va:
                best_va = vr2; best_ep = ep+1
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if STOP_EPS > 0 and vr2 >= prev_va + STOP_EPS:
                last_imp_ep = ep + 1   # 有意义进步: 重置早停钟 (噪声级抬升不计; 17.0.14)
            elif PATIENCE > 0:
                _ref = last_imp_ep if STOP_EPS > 0 else best_ep   # STOP_EPS=0 → 历史行为 (以 best_ep 为钟)
                if (ep+1) - _ref >= PATIENCE:
                    print(f'  [{name}] early stop @ ep {ep+1} (patience {PATIENCE}; best val_R^2={best_va:.4f} @ ep {best_ep})', flush=True)
                    break
    model.load_state_dict(best_sd)
    if out_ckpt:
        torch.save({'state_dict': model.state_dict(), 'num_types': NUM_TYPES, 'n_cont': n_cont,
                    'emb': EMB, 'hid': HID, 'k': K, 'dropout': DROPOUT,
                    'struct_mode': STRUCT_MODE, 'n_cont_base': N_CONT_BASE, 'cone_v2': CONE_V2,
                    'cone_feat': CONE_FEAT, 'cone_pin': CONE_PIN,
                    'val_r2': best_va, 'best_ep': best_ep}, out_ckpt)
        print(f'  [{name}] ckpt saved -> {out_ckpt} (val_R^2={best_va:.4f} @ ep {best_ep})', flush=True)
    tr2, trho, tn = eval_blocks(tr_data)
    te2, teho, ten = eval_blocks(te_data)
    print(f'  [{name}] best_val_R^2={best_va:.4f}  train R^2={tr2:.4f} / test(未见电路) R^2={te2:.4f} Spearman={teho:.4f} (n={ten})', flush=True)
    return model, te2, teho, best_va

print('\n===== V. DelayGNN 复刻 idsavg GNN (有向边消息传递) =====', flush=True)
tr_blk = assemble(tr_l, use_edges=True); va_blk = assemble(va_l, use_edges=True); te_blk = assemble(te_l, use_edges=True)
mdl, g_te, g_rho, g_bv = run_variant('V_gnn', tr_blk, va_blk, te_blk, out_ckpt=(CKPT_PATH or None))

# 无边对照 (同特征 per-node MLP, 隔离「消息传递」贡献; 本地已证明, 全量可跳过省时)
w_te2 = w_rho2 = w_bv = None
if not NO_NOGRAPH:
    print('\n===== W. 无边对照 (同特征, 无边=per-node MLP) =====', flush=True)
    w_tr = assemble(tr_l, use_edges=False); w_va = assemble(va_l, use_edges=False); w_te = assemble(te_l, use_edges=False)
    _, w_te2, w_rho2, w_bv = run_variant('V_nograph', w_tr, w_va, w_te)

# ============================================================ 汇总 + 测试按批次分桶
print('\n===== 判定 (唯一尺度 ids_avg R^2, 电路级切分, 未见电路) =====', flush=True)
print(f'  A  GBDT15 (15特征, 部署同款)  test R^2 (上表)  -- 参考: batch_v2_full 电路级 0.674')
print(f'  V  gnn           best_val R^2={g_bv:.4f}  test R^2={g_te:.4f}  Spearman={g_rho:.4f}')
if w_te2 is not None:
    print(f'  W  nograph (MLP) best_val R^2={w_bv:.4f}  test R^2={w_te2:.4f}  Spearman={w_rho2:.4f}')
else:
    print('  W  nograph (MLP) skipped (NO_NOGRAPH=1; 本地对照已证 gnn-nograph=+0.09)')

# 测试按批次分桶 (GNN 与 GBDT15 同电路同序: te_blk 按 te_l 建, gnn 预测序 = 电路内 sup 序 = gb_by_cid)
# ③ 17.0.15: 每轮常开 n_t 分层残差 R² (判定 ②' 值不值投); DUMP_PATH 非空另存 parquet 供细看
import collections
per_b = collections.defaultdict(lambda: {'p_g': [], 'p_gb': [], 't': []})
_tier = collections.defaultdict(lambda: [[], []])   # tier -> [y, pred]
_recs = []
_nb = N_CONT_BASE
with torch.no_grad():
    for cid, (ty, feat, ei, s_i, s_y) in zip([c for c in te_l if c in blocks], te_blk):
        btag = blocks[cid]['batch']
        ty = ty.to(DEV); feat = feat.to(DEV); ei = ei.to(DEV); s_i = s_i.to(DEV)
        pred = mdl(ty, feat, ei, ty.shape[0])[s_i].cpu().numpy()
        per_b[btag]['p_g'].extend(pred.tolist()); per_b[btag]['t'].extend(s_y.numpy().tolist())
        _gb = gb_by_cid.get(cid, [])
        per_b[btag]['p_gb'].extend(_gb)
        ns = blocks[cid]['ns']; N = ns.shape[0]
        s_i_cpu = s_i.cpu().numpy(); s_y_cpu = s_y.cpu().numpy()
        sup_nt = ns[s_i_cpu % N, 7]
        for _m, _k in [((sup_nt <= 2), 'n<=2'), ((sup_nt >= 3) & (sup_nt <= 7), 'n3-7'), ((sup_nt >= 8), 'n>=8')]:
            if _m.any():
                _tier[_k][0].extend(s_y_cpu[_m].tolist()); _tier[_k][1].extend(pred[_m].tolist())
        if DUMP_PATH:
            feat_cpu = feat.cpu().numpy()
            for j in range(len(s_i_cpu)):
                t = int(s_i_cpu[j]); gi = t % N
                nsr = ns[gi]; fr = feat_cpu[t]
                rec = {'circuit': cid, 'batch': btag, 'type': int(nsr[0]),
                       'log_fan': float(nsr[1]), 'depth': float(nsr[2]), 'drive': float(nsr[3]),
                       'parasitic': float(nsr[4]), 'logic_g': float(nsr[5]), 'log_h': float(nsr[6]),
                       'n_t': float(nsr[7]), 'y': float(s_y_cpu[j]), 'pred': float(pred[j]),
                       'gb_pred': (float(_gb[j]) if j < len(_gb) else float('nan')),
                       'dir_code': float(fr[_nb + 4])}
                if ns.shape[1] >= 10:
                    rec['stack'] = float(nsr[8]); rec['parallel'] = float(nsr[9])
                if CONE_N:                             # 锥体 v1/v2 (cones 从 +7 起; 无锥体 CONE_N=0 => 不读) —— 修 17.0.15 R3 越界 bug (原守卫 N_EXTRA>=7 在 N_EXTRA=7 也误触发)
                    rec['incone'] = float(fr[_nb + 7]); rec['cone_down'] = float(fr[_nb + 8]); rec['cone_up'] = float(fr[_nb + 9])
                if CONE_V2:                            # v2 专属 +10..+12 (v1 N_EXTRA=10 无, 勿读)
                    rec['main_path'] = float(fr[_nb + 10]); rec['cone_fout'] = float(fr[_nb + 11]); rec['cone_fin'] = float(fr[_nb + 12])
                if CONE_PIN:                           # ②' 三通道恒在 extras 尾部 (offset = N_EXTRA-PIN_N, 见 assemble)
                    _po = N_EXTRA - PIN_N
                    rec['leg_cnt'] = float(fr[_nb + _po + 0]); rec['leg_frac'] = float(fr[_nb + _po + 1]); rec['leg_ge2'] = float(fr[_nb + _po + 2])
                _recs.append(rec)
print('\n  --- test 按批次来源分桶 (R^2 GNN vs GBDT15) ---')
for btag, d in per_b.items():
    if len(d['t']) < 10: continue
    t = np.array(d['t']); pg = np.array(d['p_g']); pgb = np.array(d['p_gb'])
    rg = 1-np.sum((t-pg)**2)/np.sum((t-t.mean())**2)
    rg2 = 1-np.sum((t-pgb)**2)/np.sum((t-t.mean())**2)
    print(f'    {btag:16s} n={len(t):6d}  GNN R^2={rg:.4f} | GBDT15 R^2={rg2:.4f}', flush=True)
print('\n  --- ③ test 残差按 n_t 分层 (GNN R^2; 判残差是否集中高 n_t 堆叠门) ---')
for _k in ('n<=2', 'n3-7', 'n>=8'):
    y = np.array(_tier[_k][0]); p = np.array(_tier[_k][1])
    if len(y) < 20: continue
    r = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f'    {_k:6s} n={len(y):7d}  R^2={r:.4f}  (mean_y={y.mean():.3f})', flush=True)
if DUMP_PATH and _recs:
    pd.DataFrame(_recs).to_parquet(DUMP_PATH)
    print(f'\n  残差/属性 dump -> {DUMP_PATH} ({len(_recs)} test (row,门) 行)', flush=True)
