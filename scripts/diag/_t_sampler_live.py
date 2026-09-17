"""_t_sampler_live.py — 本地**真跑** 17.4.3 的解耦：开关真的换 sampler 吗？换出来的 batch 真的不一样吗？

为什么还需要这个（`_t_sampler_decouple.py` 不够）：那个自检只验了
  ① `use_grouped_sampler` 的**决策函数**（真值表），② 源码里两个站点的**文本形状**。
它**没有**验「翻开关 → 真的换了 sampler → 真的换了 batch」。两件它照不到的事：
  - `USE_GROUPED_SAMPLER` 是否真能从 `config` 导入到 `main()` 的命名空间（我上一轮是**用 grep 查
    `__all__` 不存在**来推断的 —— 那是推断，不是运行；`import *` 的遮蔽/顺序问题 grep 看不出来）。
  - 两个分支装出来的 DataLoader 是否合法、`GroupedBatchSampler` 的 import 是否会炸。

做法（沿用本仓库自检的一贯原则：**测发布出去的那段文本本身**）：
  A) 真 `exec('from config import *')` → 断言 `USE_GROUPED_SAMPLER` 真的绑上了、默认 'auto'。
  B) 从 `src/train_sweep.py` 里**按标记注释定位、抽出两个站点的那段源码**，dedent 后**直接 exec**：
     给一个 stub dataset，四种 (mode, w) 组合各跑一次，断言装出来的 sampler **类型**符合预期。
     抽的是真文本，所以这段一旦被改动，本自检立刻跟着变 —— 不存在「测试副本漂移」。
  C) **在真数据上量两个 sampler 的 batch 组成** —— 这一步会审计 17.4.2 §(2) L1 的**前提是否成立**：
     `_local_obj_feas.py` 把「原 sampler」建模成了**随机置换**，但站点 1 的 else 分支其实是
     `CircuitGroupSampler`（**整电路打包**），而站点 1 才是**默认路径**（站点 2 在离群点清洗分支里）。
     若一个 circuit 内部本就含同组行，则「随机 batch 94% 成对项为空」对默认路径**是错的**。
     ⇒ 实测：CircuitGroupSampler（含 DataLoader 的 80 行重切）vs GroupedBatchSampler vs 随机置换。

用法（本地 Windows）：python scripts/diag/_t_sampler_live.py
"""
import ast
import glob
import io
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, 'src', 'train_sweep.py')
BATCH = 80                     # config.BATCH_SIZE
KEY = ['expr', 'corner', 'switching_pin', 'direction', 'vector']
BATCHES = ['batch_v2_full', 'batch_v2_rest', 'batch_v2_m4']
RNG = np.random.default_rng(20260917)
# 17.4.2 §(2) L1 记下的「零成对 batch 占比」，用来对账（见 Part C 末尾）
L1_REC = {'batch_v2_full': 54.2, 'batch_v2_rest': 94.0, 'batch_v2_m4': 67.3}

FAIL = 0


def ok(msg):
    print(f"  OK   {msg}")


def no(msg):
    global FAIL
    FAIL += 1
    print(f"  FAIL {msg}")


raw = io.open(SRC, encoding='utf-8').read()
tree = ast.parse(raw)
_cfg = io.open(os.path.join(ROOT, 'config.py'), encoding='utf-8').read()


def extract_def(name):
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(picked) != 1:
        print(f"FAIL 抽不到唯一函数 {name}")
        sys.exit(1)
    ns = {}
    exec(compile(ast.Module(body=picked, type_ignores=[]), SRC, 'exec'), ns)
    return ns[name]


def extract_class(name):
    picked = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name]
    if len(picked) != 1:
        print(f"FAIL 抽不到唯一类 {name}（得到 {len(picked)} 个）")
        sys.exit(1)
    ns = {'Sampler': Sampler, 'np': np, 'BATCH_SIZE': BATCH}
    exec(compile(ast.Module(body=picked, type_ignores=[]), SRC, 'exec'), ns)
    return ns[name], picked[0].lineno


def extract_branch():
    """按标记注释定位，抽出站点 1 那整段 `_grouped = ... if/else` 源码（真文本）。"""
    lines = raw.splitlines()
    start = next(i for i, L in enumerate(lines)
                 if '17.4.3：采样器由 use_grouped_sampler 决定' in L)
    end = next(i for i in range(start, len(lines)) if 'val_loader = DataLoader' in lines[i])
    seg = lines[start:end]
    if 'if _grouped:' not in '\n'.join(seg) or 'else:' not in '\n'.join(seg):
        print("FAIL 抽出的站点 1 分支不含 if/else，定位失败")
        sys.exit(1)
    return textwrap.dedent('\n'.join(seg)), start + 1, end


BRANCH_SRC, BRANCH_L0, BRANCH_L1 = extract_branch()
use_grouped_sampler = extract_def('use_grouped_sampler')
CircuitGroupSampler, CGS_LINE = extract_class('CircuitGroupSampler')
from src.utils import GroupedBatchSampler   # 真的 import（会炸就当场炸）


# ---------------------------------------------------------------- Part A
print("=" * 78)
print("A) 真 exec('from config import *')：开关到底绑没绑上 main() 的命名空间")
print("=" * 78)
_ns = {}
exec('from config import *', _ns)                     # 与 train_sweep.py:8 完全同一条语句
if 'USE_GROUPED_SAMPLER' in _ns:
    ok(f"`from config import *` 后 USE_GROUPED_SAMPLER 在场 = {_ns['USE_GROUPED_SAMPLER']!r}")
else:
    no("`from config import *` **没有**导出 USE_GROUPED_SAMPLER —— main() 会 NameError！")
if _ns.get('USE_GROUPED_SAMPLER') == 'auto':
    ok("默认 'auto'（= Grouped ⟺ RANK_LOSS_W > 0，等价旧行为）")
else:
    no(f"默认不是 'auto'，是 {_ns.get('USE_GROUPED_SAMPLER')!r}")
if _ns.get('RANK_LOSS_W') == 0.0:
    ok("RANK_LOSS_W 默认仍 0.0")
else:
    no(f"RANK_LOSS_W 默认 {_ns.get('RANK_LOSS_W')!r}")
if _ns.get('BATCH_SIZE') == BATCH:
    ok(f"BATCH_SIZE = {BATCH}（与 config 一致）")
else:
    no(f"BATCH_SIZE = {_ns.get('BATCH_SIZE')!r}")

# ---------------------------------------------------------------- Part B
print()
print("=" * 78)
print(f"B) 直接 exec 站点 1 的真源码（src/train_sweep.py:{BRANCH_L0}-{BRANCH_L1-1}）")
print("=" * 78)
print(textwrap.indent(BRANCH_SRC, '  | '))


class _StubDS:
    """只提供两个分支各自需要的东西：GroupedBatchSampler 要 .group_ids，
    CircuitGroupSampler 要 .dynamic_df['circuit_id']，DataLoader 要 __len__/__getitem__。"""
    def __init__(self, cids, gids):
        self.dynamic_df = pd.DataFrame({'circuit_id': cids})
        self.group_ids = list(gids)
        self.n = len(cids)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([float(i)])


_cids = ['c%d' % (i // 30) for i in range(600)]           # 30 行/电路
_gids = [i % 37 for i in range(600)]                      # 组随意，只要类型对
_stub = _StubDS(_cids, _gids)

COMBOS = [
    ('auto', 0.0, CircuitGroupSampler, "旧默认 = 原路径"),
    ('auto', 0.5, GroupedBatchSampler, "旧 w>0 = Grouped"),
    ('1', 0.0, GroupedBatchSampler, "**新 C 臂**（17.4.3 前无法表达）"),
    ('0', 0.5, CircuitGroupSampler, "**取消 §13.6 混淆**（w>0 但走原 sampler）"),
    ('1', 0.5, GroupedBatchSampler, "= 已跑过的 v2nowaver42m4（B 臂）"),
    ('0', 0.0, CircuitGroupSampler, "双关"),
]
print("  六种组合各 exec 一次真源码：")
_types = {}
for mode, w, want, why in COMBOS:
    ns = {'use_grouped_sampler': use_grouped_sampler,
          'RANK_LOSS_W': w, 'USE_GROUPED_SAMPLER': mode,
          'BATCH_SIZE': BATCH, 'DataLoader': DataLoader,
          'CircuitGroupSampler': CircuitGroupSampler, 'train_dataset': _stub}
    try:
        exec(compile(BRANCH_SRC, SRC, 'exec'), ns)
    except Exception as e:
        no(f"mode={mode!r} w={w} exec 抛异常：{type(e).__name__}: {e}")
        continue
    s = ns['sampler']
    _types[(mode, w)] = type(s).__name__
    got_grouped = isinstance(s, GroupedBatchSampler)
    want_grouped = (want is GroupedBatchSampler)
    if got_grouped is want_grouped:
        # 顺带验「装出来的 loader 用的哪个参数」——两类 sampler 的接法不同，不能混：
        #   Grouped 走 batch_sampler= → loader.batch_sampler **就是**该对象本身
        #   原路径走 sampler=+batch_size= → torch 会**包一层** BatchSampler，须拆开看 .sampler
        # ⚠ 首版这里写的是 `bm is None`，于是三个 else 组合全报 FAIL —— 是**自检自己写错**，
        #   不是代码错。这正是「必须真跑一次」的理由：单元自检照不到这个。
        bm = ns['train_loader'].batch_sampler
        if isinstance(s, GroupedBatchSampler):
            good = (bm is s)
            detail = f"batch_sampler= 直通（bm is sampler = {bm is s}）"
        else:
            good = isinstance(bm, torch.utils.data.BatchSampler) and bm.sampler is s
            detail = (f"sampler= 被包成 BatchSampler（bm.sampler is sampler = "
                      f"{getattr(bm, 'sampler', None) is s}，batch_size = "
                      f"{getattr(bm, 'batch_size', None)}）")
        if good:
            ok(f"mode={mode!r:<6} w={w:<4g} → {type(s).__name__:<20} {why}")
            print(f"         └ {detail}")
        else:
            no(f"mode={mode!r} w={w} sampler 类型对，但 DataLoader 接法不对：{detail}")
    else:
        no(f"mode={mode!r} w={w} 应为 {want.__name__}，实得 {type(s).__name__}")

print()
if _types.get(('auto', 0.0)) == _types.get(('0', 0.5)) == 'CircuitGroupSampler' \
        and _types.get(('1', 0.0)) == 'GroupedBatchSampler':
    ok("**解耦成立：'1' 且 w=0 得到 Grouped，而 '0' 且 w>0 得到原 sampler**")
else:
    no(f"解耦不成立：{_types}")

# ---------------------------------------------------------------- Part C
print()
print("=" * 78)
print("C) 真数据审计：两个 sampler 的 batch 组成（含对 L1「随机置换」前提的检验）")
print("=" * 78)
print(f"   站点 1 的 else 是 CircuitGroupSampler（整电路打包）= 默认路径；")
print(f"   L1 却把「原 sampler」建模成随机置换。这里实测三者的每 batch 同组成对数。")
print(f"   注：CircuitGroupSampler 以 sampler= 传入 ⇒ DataLoader **再按 {BATCH} 连续重切**，")
print(f"       故这里照做（否则会把「电路完整性」误当成「组完整性」）。")


def load_codes(b):
    parts = sorted(glob.glob(os.path.join(ROOT, 'data', b, 'timing_arcs*.parquet')))
    if not parts:
        return None
    cols = ['DELAY', 'circuit_id'] + KEY
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in parts], ignore_index=True) \
        if len(parts) > 1 else pd.read_parquet(parts[0], columns=cols)
    df = df.dropna(subset=['DELAY'])
    df = df[df['DELAY'] > 1e-12].reset_index(drop=True)
    gk = None
    for name in KEY:
        c = df[name].astype(str).to_numpy() if name in df.columns \
            else np.full(len(df), '', dtype=object)
        gk = c if gk is None else np.char.add(np.char.add(gk, '|'), c)
    gid = pd.factorize(gk)[0]
    cid = pd.factorize(df['circuit_id'].astype(str).to_numpy())[0]
    return cid, gid


def pairs_in_batches(batches):
    """每 batch 的同组有序对数（只数 j>i 同组）。"""
    out = []
    for b in batches:
        s = np.sort(np.asarray(b))
        if s.size < 2:
            out.append(0); continue
        n = 0
        i = 0
        while i < s.size:
            j = i + 1
            while j < s.size and s[j] == s[i]:
                j += 1
            m = j - i
            n += m * (m - 1) // 2
            i = j
        out.append(n)
    return np.array(out, dtype=np.int64)


def chunk(lst, k):
    return [lst[i:i + k] for i in range(0, len(lst), k)]


for b in BATCHES:
    got = load_codes(b)
    if got is None:
        print(f"\n[{b}] 缺文件，跳过")
        continue
    cid, gid = got
    n = len(gid)
    print(f"\n[{b}] 行 {n:,}｜电路 {cid.max()+1:,}｜组 {gid.max()+1:,}")

    # (1) CircuitGroupSampler：真类、真 __iter__、再按 80 重切（= 站点 1 的实际语义）
    ds = _StubDS(['c%d' % c for c in cid], list(gid))
    np.random.seed(20260917)
    flat = list(iter(CircuitGroupSampler(ds)))
    b_circ = chunk(flat, BATCH)
    # 该 sampler 会把 index 列表复制到 n_samples，可能引入重复 → 说明清楚
    p_circ = pairs_in_batches([gid[np.asarray(x)] for x in b_circ])

    # (2) GroupedBatchSampler：真类，batch_sampler 语义（不再重切）
    p_grp = pairs_in_batches([gid[np.asarray(x)] for x in
                              GroupedBatchSampler(list(gid), BATCH, shuffle=True)])

    # (3) L1 的建模：随机置换 + 80 chunks。
    # ⚠ 必须多抽几个置换：单次置换的统计量本身有抖动（m4 只有 ~1.4k 个 batch，
    #   L1 的脚本里 RNG 流被 l1() 的第二处 permutation 消耗过 ⇒ 抽到的是**另一个**置换，
    #   于是 69.8% vs L1 记的 67.3%。那不是「L1 算错」，是**同一个量的抽样抖动** ——
    #   所以这里报 mean 与 min~max 区间，并把 L1 的值拿去**判是否落在区间内**。
    _rnd_pcts = []
    for _ in range(20):
        perm = RNG.permutation(n)[: (n // BATCH) * BATCH]
        _p = pairs_in_batches([gid[perm[i:i + BATCH]] for i in range(0, len(perm), BATCH)])
        _rnd_pcts.append((100.0 * (_p == 0).mean(), _p.mean()))
    p_rnd = _p                                   # 最后一次，供打印
    _rnd_zero_mean = float(np.mean([z for z, _ in _rnd_pcts]))
    _rnd_zero_lo = min(z for z, _ in _rnd_pcts)
    _rnd_zero_hi = max(z for z, _ in _rnd_pcts)

    for tag, p in [('CircuitGroupSampler(默认路径)', p_circ),
                   ('随机置换(= L1 的建模)', p_rnd),
                   ('GroupedBatchSampler(C 臂)', p_grp)]:
        print(f"   {tag:<30} batch {len(p):>6,}｜零成对占比 **{100.0*(p==0).mean():6.1f}%**"
              f"｜成对均值 {p.mean():9.2f}｜中位 {int(np.median(p)):>5}")
    r = p_grp.mean() / max(p_circ.mean(), 1e-9)
    print(f"   ⇒ Grouped / Circuit 的成对均值之比 = **{r:,.1f}×**")

    # 与 L1 记录的读数对账
    _l1 = L1_REC.get(b)
    _def_pct = 100.0 * (p_circ == 0).mean()
    if _l1 is not None:
        if abs(_rnd_zero_mean - _l1) <= 3.0:
            ok(f"随机置换 20 次均值 {_rnd_zero_mean:.1f}%（{_rnd_zero_lo:.1f}~{_rnd_zero_hi:.1f}%）"
               f"与 L1 记的 {_l1}% 差 {_rnd_zero_mean - _l1:+.1f}pp ≤3pp ⇒ "
               f"L1 脚本没算错，差的是**抽样抖动**（该量本身就是个有抖动的统计量）")
        else:
            no(f"随机置换均值 {_rnd_zero_mean:.1f}% 与 L1 的 {_l1}% 差 >3pp ⇒ 复现失败，先查脚本")
        _inside = min(_def_pct, 100.0) < _rnd_zero_lo or _def_pct > _rnd_zero_hi
        print(f"   ⚠ **默认路径**（CircuitGroupSampler）实测 **{_def_pct:.1f}%**，"
              f"与 L1 记录 {_l1}% 差 **{_def_pct - _l1:+.1f}pp**"
              f"{' —— 落在随机置换区间之外' if _inside else ''}")
        print(f"      ⇒ L1 那个数**测的不是实际跑的 sampler**：它建模的是站点 2 的 else"
              f"（`shuffle=True`），而站点 1 的 else 是 CircuitGroupSampler")

print()
print("=" * 78)
if FAIL:
    print(f"FAIL 共 {FAIL} 项")
    sys.exit(1)
print("OK 全部通过")
