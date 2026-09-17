"""_t_sampler_decouple.py — 本地自检 17.4.3 的「采样器 与 RANK_LOSS_W 解耦」（不需要服务器/训练/torch）。

要守住的核心不变量只有一条：**默认路径（'auto'）下的采样器选择必须与 17.4.3 之前逐字相同**。
一旦这条破了，所有已跑过的 run 都不可复现 —— 而这次改动的全部价值就在于
「能把采样器单独打开，以便把 §13.6 那次否证拆成 采样器效应 / 损失效应 两半」。

做三件事：
  ① 从 src/train_sweep.py **抽出** use_grouped_sampler 直接跑真值表（测的是发布出去的那段文本本身）；
  ② 源码守卫：两个站点都走该函数、残余的 `RANK_LOSS_W > 0` 只剩损失那一处、
     GroupedBatchSampler 仍只构造两次、两个 else 分支逐字未动；
  ③ config.py 守卫：默认 'auto'（等价旧行为）。

用法（本地 Windows）：python scripts/diag/_t_sampler_decouple.py

⚠ **本脚本只覆盖「决策函数 + 源码形状」，不覆盖「真跑」** —— 「翻开关是否真的换了 sampler、
   换出来的 batch 是否真的不同、`from config import *` 是否真能导出该常量」在
   **`scripts/diag/_t_sampler_live.py`**（真 exec 站点源码 + 真数据量 batch 组成）。
   两个都要跑；只跑本脚本会漏掉真跑才照得到的问题。
"""
import ast
import os
import re
import sys

# 控制台默认是 GBK，打不出 ⚠/✓ → 强制 UTF-8（否则诊断信息自己会崩，掩盖真实结论）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'src', 'train_sweep.py')
CFG = os.path.join(ROOT, 'config.py')

FAIL = 0


def ok(msg):
    print(f"  OK   {msg}")


def no(msg):
    global FAIL
    FAIL += 1
    print(f"  FAIL {msg}")


# ---------- ① 从真源码里抽出 use_grouped_sampler ----------
def load_func(name):
    text = open(SRC, encoding='utf-8').read()
    tree = ast.parse(text)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(picked) != 1:
        print(f"FAIL 抽不到唯一函数 {name}（得到 {len(picked)} 个）")
        sys.exit(1)
    ns = {}
    exec(compile(ast.Module(body=picked, type_ignores=[]), SRC, 'exec'), ns)
    return ns[name]


use_grouped_sampler = load_func('use_grouped_sampler')
print(f"抽取到 use_grouped_sampler（源: {os.path.relpath(SRC, ROOT)}）")

raw = open(SRC, encoding='utf-8').read()
cfg_raw = open(CFG, encoding='utf-8').read()


# ---------- ② 真值表 ----------
print("--- 1) 'auto'：必须与 17.4.3 之前逐字等价（Grouped ⟺ RANK_LOSS_W > 0）---")
CASES_AUTO = [
    (0.0, 'auto', False, "w=0 → 原 sampler（旧 `if RANK_LOSS_W > 0` 走 else）"),
    (0.5, 'auto', True, "w>0 → Grouped（旧代码走 if）"),
    (1e-9, 'auto', True, "w 极小但 >0 → Grouped（与旧代码的严格 > 一致）"),
    (-1.0, 'auto', False, "w<0 → 原 sampler（旧代码同样走 else，别自作聪明报错）"),
]
for w, mode, want, why in CASES_AUTO:
    got = use_grouped_sampler(w, mode)
    if got is want:
        ok(f"auto, w={w:<7g} → {got!s:<5}  {why}")
    else:
        no(f"auto, w={w} 应为 {want}，得到 {got}")

print("--- 2) 显式开关：与 RANK_LOSS_W **无关**（这正是解耦本身）---")
for mode, want in [('1', True), ('0', False), ('true', True), ('false', False),
                   ('yes', True), ('no', False), ('on', True), ('off', False)]:
    for w in (0.0, 0.5):
        got = use_grouped_sampler(w, mode)
        if got is want:
            ok(f"mode={mode!r:<8} w={w:<5g} → {got!s:<5}（与 w 无关）")
        else:
            no(f"mode={mode!r} w={w} 应为 {want}，得到 {got}")
# 关键两组：w=0 却开 Grouped（新臂 C）、w>0 却关 Grouped（取消 §13.6 的混淆）
if use_grouped_sampler(0.0, '1') is True:
    ok("**w=0 且 USE_GROUPED_SAMPLER=1 → Grouped** ⟸ 这三臂实验的 C 臂，17.4.3 前无法表达")
else:
    no("w=0 + '1' 必须为 Grouped（C 臂做不出来）")
if use_grouped_sampler(0.5, '0') is False:
    ok("**w>0 且 USE_GROUPED_SAMPLER=0 → 原 sampler** ⟸ 拆开 §13.6 混淆用")
else:
    no("w>0 + '0' 必须为原 sampler")

print("--- 3) 大小写/空白容错 + 未知值回退到 'auto' ---")
for mode in ('1 ', ' 1', 'TRUE', 'True', 'Yes', 'ON', '\t0\n', 'OFF'):
    want = str(mode).strip().lower() in ('1', 'true', 'yes', 'on')
    got = use_grouped_sampler(0.0, mode)
    if got is want:
        ok(f"mode={mode!r} → {got!s}（strip+lower 生效）")
    else:
        no(f"mode={mode!r} 应为 {want}，得到 {got}")
for mode in ('auto', '', 'none', 'maybe', 'grouped'):
    # 未知值一律回退 = 按旧行为（w=0 → False，w=0.5 → True）
    if use_grouped_sampler(0.0, mode) is False and use_grouped_sampler(0.5, mode) is True:
        ok(f"mode={mode!r} → 回退按 w（未知值不报错、不静默改行为）")
    else:
        no(f"mode={mode!r} 未正确回退到 auto")

print("--- 4) 纯函数性质：同输入同输出、不抛异常 ---")
if all(use_grouped_sampler(0.3, 'auto') == use_grouped_sampler(0.3, 'auto') for _ in range(3)):
    ok("无隐藏状态（同输入同输出）")
else:
    no("非确定性")
try:
    use_grouped_sampler(None, 'auto')          # 传 None 不该崩（应在 float() 处回退/报错要明确）
    no("w=None + 'auto' 竟然没报错 —— 静默通过比报错更危险")
except (TypeError, ValueError):
    ok("w=None + 'auto' 明确报错（不静默当成 False）")

# ---------- ③ 源码守卫 ----------
print("--- 5) 源码守卫：两站点都解耦了，且默认路径未被动过 ---")
# ⚠ 必须用 AST 计数、不能用正则：本文件的文档字符串/打印串里也出现 `GroupedBatchSampler(` 和
#   `RANK_LOSS_W > 0` 这些字样（正是为了说明白），正则会把它们算成真实代码 → 自检自己骗自己。
#   （首版就是这么写的，报出「GroupedBatchSampler 构造=4」这种假阳性。）
_tree = ast.parse(raw)


def calls_of(name):
    return [n for n in ast.walk(_tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


_calls_ugs = calls_of('use_grouped_sampler')
n_def = len([n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == 'use_grouped_sampler'])
_calls_grp = calls_of('GroupedBatchSampler')
_cmps_rlw = [n for n in ast.walk(_tree)
             if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
             and n.left.id == 'RANK_LOSS_W']
n_circ = len(re.findall(r'CircuitGroupSampler\(train_dataset\)', raw))
n_shuf = len(re.findall(r'train_loader = DataLoader\(train_subset, batch_size=BATCH_SIZE, shuffle=True', raw))
print(f"      use_grouped_sampler 调用={len(_calls_ugs)}（行 {[n.lineno for n in _calls_ugs]}）"
      f"  真·GroupedBatchSampler 构造={len(_calls_grp)}"
      f"  RANK_LOSS_W 比较={len(_cmps_rlw)}（行 {[n.lineno for n in _cmps_rlw]}）"
      f"  CircuitGroupSampler(train_dataset)={n_circ}  站点2 else={n_shuf}")

if len(_calls_ugs) == 2:
    ok(f"两个站点都用 use_grouped_sampler（行 {[n.lineno for n in _calls_ugs]}，无一漏改）")
else:
    no(f"应有 2 处解耦调用，得到 {len(_calls_ugs)}（会一半解耦一半没解耦 = 更隐蔽的坑）")
if len(_calls_ugs) == 2 and all(
        len(n.args) == 2 and getattr(n.args[1], 'id', None) == 'USE_GROUPED_SAMPLER' for n in _calls_ugs):
    ok("两处都把 USE_GROUPED_SAMPLER 当第 2 参传入（不是漏传、也不是写死 'auto'）")
else:
    no("有调用点没传 USE_GROUPED_SAMPLER（写死 'auto' 等于开关失效）")
if n_def == 1:
    ok("use_grouped_sampler 只定义一次（是模块级纯函数，可被 ast 抽出）")
else:
    no(f"应有 1 个定义，得到 {n_def}")
if len(_cmps_rlw) == 1:
    _c = _cmps_rlw[0]
    if isinstance(_c.ops[0], ast.Gt) and getattr(_c.comparators[0], 'value', None) == 0:
        ok(f"残余 `RANK_LOSS_W > 0` 只剩 1 处（行 {_c.lineno}，即损失开关本身，形式仍是 > 0）")
    else:
        no(f"损失开关的比较形式变了：行 {_c.lineno}")
else:
    no(f"RANK_LOSS_W 比较应有 1 处（损失开关），得到 {len(_cmps_rlw)} "
       f"（行 {[n.lineno for n in _cmps_rlw]}）—— 采样器可能还挂在 RANK_LOSS_W 上")
if len(_calls_grp) == 2:
    ok(f"GroupedBatchSampler 真构造仍只 2 次（行 {[n.lineno for n in _calls_grp]}，两站点各一）")
else:
    no(f"GroupedBatchSampler 构造点应为 2，得到 {len(_calls_grp)}")
if n_circ == 1:
    ok("站点 1 else 分支逐字未动（CircuitGroupSampler(train_dataset)）")
else:
    no(f"站点 1 的 else 分支被改动/丢失（找到 {n_circ} 处）")
if n_shuf == 1:
    ok("站点 2 else 分支逐字未动（shuffle=True，与站点 1 的 else **本来就不同**）")
else:
    no(f"站点 2 的 else 分支被改动/丢失（找到 {n_shuf} 处）—— 注意它与站点 1 的 else 不一样")
if len(re.findall(r'use_grouped_sampler\(OUT', raw)) == 0:
    ok("没有把 rank_loss_w 以外的参数顺序写反的调用")
else:
    no("调用签名疑似写错")

# ---------- ④ config 守卫 ----------
print("--- 6) config.py 守卫：默认必须等价旧行为 ---")
m = re.search(r"^USE_GROUPED_SAMPLER = os\.environ\.get\('USE_GROUPED_SAMPLER', '([^']*)'\)",
              cfg_raw, re.M)
if m and m.group(1) == 'auto':
    ok("USE_GROUPED_SAMPLER 默认 'auto'（= 逐字不改现有一切 run）")
else:
    no(f"默认值应为 'auto'，得到 {m.group(1) if m else '未找到该行'}")
m = re.search(r"^RANK_LOSS_W = ([0-9.eE+-]+)", cfg_raw, re.M)
if m and float(m.group(1)) == 0.0:
    ok("RANK_LOSS_W 默认仍为 0.0（损失未启用）")
else:
    no(f"RANK_LOSS_W 默认应为 0.0，得到 {m.group(1) if m else '未找到'}")

print("==================================================")
if FAIL:
    print(f"FAIL 共 {FAIL} 项")
    sys.exit(1)
print("OK 全部通过")
