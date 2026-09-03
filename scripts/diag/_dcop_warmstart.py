#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_dcop_warmstart.py — DCOP warm-start 可行性机制测试（Phase A，单 deck 自暖 = 天花板上限）

背景：Rust 侧每个候选跑一次全电路 Xyce（.tran 1p 1n + delay measure）。16.11.1 已证
  DCOP 是固定开销大头（小电路 ~0.07s/0.19s，~37%），且 reltol 不碰 DCOP。本脚本测两种
  让 DCOP "暖启动"的机制能否省下这块，且延迟不变：

  M1  .nodeset <t0电压>          —— 保留 DCOP，给它近解初值 → 减少牛顿迭代。
                                   收敛点仍是真实 DCOP → 延迟理论上逐位不变。
  M2  .ic <t0电压> + .tran..uic  —— 跳过 DCOP，直接从基线 DCOP 电压起瞬态。
                                   （区别已失败的 NOOP：NOOP 从 0 起改物理；M2 从正确
                                   工作点起，近似一致。）

关键便利：Rust 模板自带 `.print tran V(*)`，Xyce `.prn` 的 t=0 行 = DCOP 节点电压，
  无需任何额外导出——warm 值直接读 cold run 自己的 .prn。

单 deck 自暖 = 用"自己精确的 t=0"当 warm 值，测的是**机制天花板**：
  - 若完美 warm 都不省时间 → 方向死（便宜否决，不用碰 Rust 代码）；
  - 若省且延迟逐位一致 → 再上"近孪生基线暖候选"（生产形态，测真实差距 + 决策轨迹）。

用法（server，Xyce 环境已就绪）：
  python3 _dcop_warmstart.py --deck /abs/path/to/a_deck.sp [--xyce /path/to/Xyce] [--repeats 3]
  （--deck 可多次给，逐个跑；deck 需含 delay measure 与 .print tran V(*)，Rust 生成的都是。）

输出：每个 deck 一张表：cold / M1(nodeset) / M2(ic+uic) 的 中位墙钟 + delay + 延迟相对误差；
  另打印 .prn 结构探针（节点数、头几行），任何变体失败会把 Xyce stderr 落盘并标 ERROR。
"""
import argparse
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time


def read_text(p):
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


# ---------- 1) deck 准备 ----------
def make_deck(src, work_dir, tag, ic_lines=(), nodeset_lines=(), uic=False):
    """复制 deck 到 work_dir/<tag>_<i>/，保证 .print tran V(*)、.tran 行、插 .ic/.nodeset。
    返回 (deck_sp 路径)。不改 src。"""
    d = os.path.join(work_dir, f"{tag}_{os.path.basename(src)}")
    os.makedirs(d, exist_ok=True)
    deck = os.path.join(d, os.path.basename(src))
    shutil.copy2(src, deck)
    content = read_text(deck)

    # .include 相对路径 -> 相对新位置（若 include 的是同目录相对路径则失效，改成绝对 src 目录）
    content = re.sub(r'(\.include\s+)"([^"]+)"', lambda m: f'.include "{m.group(2)}"', content)

    # 确保有 .print tran V(*)（warm 值来源）；没有就在 .end 前插
    if not re.search(r'(?im)^\.print\s+tran\s+V\(\*\)', content):
        content = re.sub(r'(?im)^\.end\s*$',
                         '.print tran V(*)\n.end', content, count=1)

    # .tran 行（uic 变体追加 uic）
    if uic:
        content = re.sub(r'(?im)^\.tran\s+[^\n]+',
                         lambda m: m.group(0).rstrip() + ' uic', content, count=1)

    # 插 .nodeset / .ic 到 .end 前
    extras = list(nodeset_lines) + list(ic_lines)
    if extras:
        block = "\n".join(extras) + "\n"
        content = re.sub(r'(?im)^\.end\s*$', block + '.end', content, count=1)

    write_text(deck, content)
    return deck


# ---------- 2) 跑 Xyce ----------
def run_xyce(deck, xyce, work_dir, tag, timeout_s=120):
    """在 deck 所在目录跑 Xyce，返回 (ok, wall_s, stdout, stderr, mt0, prn)。"""
    cwd = os.path.dirname(deck)
    mt0 = deck + ".mt0"
    prn = deck + ".prn"
    for p in (mt0, prn):
        if os.path.exists(p):
            os.remove(p)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([xyce, os.path.basename(deck)],
                              cwd=cwd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, float("nan"), "", "TIMEOUT", None, None
    wall = time.perf_counter() - t0
    return (proc.returncode == 0), wall, proc.stdout, proc.stderr, mt0, prn


# ---------- 3) 解析 ----------
def parse_delay(mt0):
    """从 .mt0 取 delay 值（若有）。返回 float 或 None。"""
    if not mt0 or not os.path.exists(mt0):
        return None
    for line in read_text(mt0).splitlines():
        if line.lower().startswith("delay"):
            parts = line.split()
            for p in parts[1:]:
                try:
                    return float(p)
                except ValueError:
                    continue
    return None


def parse_t0_voltages(prn, max_nodes=4000):
    """从 .prn 解析 t≈0 行所有 V(*) 节点电压。返回 [(node_name, volt)]。
    Xyce .prn 表头含 'Index'/'time' 与 'V(name)' 列；取首条数值行（t=0 即 DCOP）。
    """
    if not prn or not os.path.exists(prn):
        return None, []
    lines = read_text(prn).splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if len(toks) >= 2 and toks[0].lower() == "index" and "time" in [t.lower() for t in toks[:3]]:
            header_idx = i
            break
    if header_idx is None:
        return None, []
    headers = lines[header_idx].split()
    # 找首个数值行（index=0）
    for ln in lines[header_idx + 1:]:
        toks = ln.split()
        if len(toks) >= 2:
            try:
                idx = int(toks[0])
            except ValueError:
                continue
            if idx == 0:
                pairs = []
                for col, h in enumerate(headers):
                    if col < 2:
                        continue
                    m = re.match(r'^V\((.*)\)$', h.strip())
                    if m and col < len(toks):
                        name = m.group(1)
                        try:
                            v = float(toks[col])
                        except ValueError:
                            continue
                        if name:
                            pairs.append((name, v))
                return (idx, pairs[:max_nodes])
    return None, []


def median_times(xyce, deck, repeats):
    """跑 repeats 次，返回中位墙钟；任何一次失败返回 nan。"""
    wall = []
    for _ in range(repeats):
        ok, w, _, _, _, _ = run_xyce(deck, xyce, os.path.dirname(deck), "m")
        if not ok:
            return float("nan")
        wall.append(w)
    return statistics.median(wall)


# ---------- 4) 主流程 ----------
def probe_deck(deck):
    c = read_text(deck)
    has_print = bool(re.search(r'(?im)^\.print\s+tran\s+V\(\*\)', c))
    tran = re.search(r'(?im)^\.tran\s+([^\n]+)', c)
    print(f"  [probe] {deck}\n    .print V(*): {has_print} | .tran: {tran.group(1).strip() if tran else 'N/A'}")


def run_one(src, xyce, work_dir, repeats, timeout_s):
    print(f"\n===== deck: {src} =====")
    probe_deck(src)

    # cold（原始 deck 自跑，不改）
    cold_deck = make_deck(src, work_dir, "cold")
    ok, w, out, err, mt0, prn = run_xyce(cold_deck, xyce, work_dir, "cold", timeout_s)
    if not ok:
        print(f"  ERROR cold 跑不起来（stderr 见 {cold_deck}.err）。终止该 deck。")
        write_text(cold_deck + ".err", err or "")
        return
    delay_cold = parse_delay(mt0)
    t0_info, volts = parse_t0_voltages(prn)
    print(f"  cold ok wall={w:.4f}s delay={delay_cold} | t0 节点数={len(volts)} "
          f"({t0_info}) 样例={volts[:2] if volts else '无'}")

    if not volts:
        print("  ERROR 解析不到 t=0 电压，无法 warm。看 .prn 头几行：")
        if prn and os.path.exists(prn):
            for ln in read_text(prn).splitlines()[:6]:
                print("    |", ln)
        return

    # 中位墙钟（多次）
    tcold = median_times(xyce, cold_deck, repeats)
    print(f"  cold  中位墙钟 = {tcold:.4f}s (repeats={repeats})")

    # 构造 warm 行
    nodeset_lines = [f".nodeset V({n})={v:.9g}" for n, v in volts]
    ic_lines = [f".ic V({n})={v:.9g}" for n, v in volts]

    # ---- M1 nodeset ----
    m1 = make_deck(src, work_dir, "m1nodeset", nodeset_lines=nodeset_lines)
    ok1, w1, _, err1, mt1, _ = run_xyce(m1, xyce, work_dir, "m1", timeout_s)
    if not ok1:
        print(f"  M1 ERROR: Xyce 退非零（.nodeset 名不识别?）。stderr 见 {m1}.err")
        write_text(m1 + ".err", err1 or "")
        d1, t1 = None, float("nan")
    else:
        t1 = median_times(xyce, m1, repeats)
        d1 = parse_delay(mt1)
    e1 = "—" if (delay_cold is None or d1 is None) else f"{abs(d1-delay_cold)/abs(delay_cold)*100:.4f}%"

    # ---- M2 ic + uic ----
    m2 = make_deck(src, work_dir, "m2icuic", ic_lines=ic_lines, uic=True)
    ok2, w2, _, err2, mt2, _ = run_xyce(m2, xyce, work_dir, "m2", timeout_s)
    if not ok2:
        print(f"  M2 ERROR: Xyce 退非零（.ic 名不识别 / uic 语法?）。stderr 见 {m2}.err")
        write_text(m2 + ".err", err2 or "")
        d2, t2 = None, float("nan")
    else:
        t2 = median_times(xyce, m2, repeats)
        d2 = parse_delay(mt2)
    e2 = "—" if (delay_cold is None or d2 is None) else f"{abs(d2-delay_cold)/abs(delay_cold)*100:.4f}%"

    save1 = (tcold - t1) / tcold * 100 if tcold and not (tcold != tcold or t1 != t1) else float("nan")
    save2 = (tcold - t2) / tcold * 100 if tcold and not (tcold != tcold or t2 != t2) else float("nan")

    print("\n  结果表：")
    print(f"    {'变体':<14}{'中位墙钟(s)':<14}{'vs cold 节省%':<14}{'delay':<14}{'延迟误差'}")
    print(f"    {'cold':<14}{tcold:<14.4f}{'—':<14}{delay_cold!s:<14}{'—'}")
    print(f"    {'M1 .nodeset':<14}{t1:<14.4f}{save1:<14.4f}{str(d1):<14}{e1}")
    print(f"    {'M2 .ic+uic':<14}{t2:<14.4f}{save2:<14.4f}{str(d2):<14}{e2}")
    print(f"    （基准：16.11.1 reltol 约 1.2-1.4x≈省 17-29%；DCOP 占小电路 ~37%。）")


def run_one_interleaved(src, xyce, work_dir, repeats, timeout_s):
    """交错版：每轮依次 cold / M1 / M2 各跑一次（同轮内比值消负载漂移），
    打印每轮原始墙钟 + 轮内比值中位数。用于判断『波动是负载还是真实效应』。"""
    print(f"\n===== deck (interleaved): {src} =====")
    probe_deck(src)
    cold_deck = make_deck(src, work_dir, "cold")
    ok, _, out, err, mt0, prn = run_xyce(cold_deck, xyce, work_dir, "cold", timeout_s)
    if not ok:
        print(f"  ERROR cold 跑不起来（stderr 见 {cold_deck}.err）")
        write_text(cold_deck + ".err", err or "")
        return
    delay_cold = parse_delay(mt0)
    _, volts = parse_t0_voltages(prn)
    print(f"  cold ok delay={delay_cold} | t0 节点数={len(volts)}")
    if not volts:
        print("  ERROR 解析不到 t=0 电压，无法 warm。")
        return
    nodeset_lines = [f".nodeset V({n})={v:.9g}" for n, v in volts]
    ic_lines = [f".ic V({n})={v:.9g}" for n, v in volts]
    m1 = make_deck(src, work_dir, "m1nodeset", nodeset_lines=nodeset_lines)
    m2 = make_deck(src, work_dir, "m2icuic", ic_lines=ic_lines, uic=True)
    # 先各 probe 一次确认能跑，取 delay
    ok1, _, _, err1, mt1, _ = run_xyce(m1, xyce, work_dir, "m1", timeout_s)
    if not ok1:
        print(f"  M1 ERROR: 见 {m1}.err"); write_text(m1 + ".err", err1 or ""); return
    ok2, _, _, err2, mt2, _ = run_xyce(m2, xyce, work_dir, "m2", timeout_s)
    if not ok2:
        print(f"  M2 ERROR: 见 {m2}.err"); write_text(m2 + ".err", err2 or ""); return
    d1, d2 = parse_delay(mt1), parse_delay(mt2)

    # 轮内交错
    rows = []  # (r, cold, m1, m2)
    for r in range(repeats):
        ws = {}
        for name, deck in (("cold", cold_deck), ("m1", m1), ("m2", m2)):
            okw, w, _, _, _, _ = run_xyce(deck, xyce, work_dir, name, timeout_s)
            ws[name] = w if okw else float("nan")
        rows.append((r, ws["cold"], ws["m1"], ws["m2"]))
        print(f"  round {r}: cold={ws['cold']:.4f}s  m1={ws['m1']:.4f}s  "
              f"m2={ws['m2']:.4f}s  | m1/cold={ws['m1']/ws['cold']:.3f}  m2/cold={ws['m2']/ws['cold']:.3f}")

    def med(key, idx):
        vals = [row[idx] for row in key]
        vals = [v for v in vals if v == v]
        return statistics.median(vals) if vals else float("nan")
    t_c, t1, t2 = med(rows, 1), med(rows, 2), med(rows, 3)
    rm1 = statistics.median([r[2] / r[1] for r in rows])
    rm2 = statistics.median([r[3] / r[1] for r in rows])
    e1 = "—" if (delay_cold is None or d1 is None) else f"{abs(d1-delay_cold)/abs(delay_cold)*100:.4f}%"
    e2 = "—" if (delay_cold is None or d2 is None) else f"{abs(d2-delay_cold)/abs(delay_cold)*100:.4f}%"
    print("\n  结果表（轮内比值中位）：")
    print(f"    {'变体':<10}{'中位墙钟':<12}{'vs cold(轮内比值)':<20}{'delay':<14}{'误差'}")
    print(f"    {'cold':<10}{t_c:<12.4f}{'1.000':<20}{delay_cold!s:<14}{'—'}")
    print(f"    {'M1':<10}{t1:<12.4f}{(rm1-1)*100:<19.3f}%{str(d1):<14}{e1}")
    print(f"    {'M2':<10}{t2:<12.4f}{(rm2-1)*100:<19.3f}%{str(d2):<14}{e2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", action="append", required=True,
                    help="Rust 生成的 tb deck 路径（含 delay measure + .print tran V(*)）")
    ap.add_argument("--xyce", default="/usr/local/bin/Xyce")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--interleave", action="store_true",
                    help="交错测 cold/M1/M2（每轮各一次），比值消负载漂移，判波动是否真实")
    ap.add_argument("--timeout-s", type=float, default=120)
    ap.add_argument("--work", default=None, help="work 目录（默认 tempfile）")
    args = ap.parse_args()

    work_dir = args.work or tempfile.mkdtemp(prefix="dcopwarm_")
    print(f"work_dir = {work_dir}")
    for dk in args.deck:
        if args.interleave:
            run_one_interleaved(os.path.abspath(dk), args.xyce, work_dir, args.repeats, args.timeout_s)
        else:
            run_one(os.path.abspath(dk), args.xyce, work_dir, args.repeats, args.timeout_s)


if __name__ == "__main__":
    main()
