"""_t_rank_probe.py — 同一批候选，两种聚合口径对着同一把真值尺子（本地临时，不入库）。

**为什么需要它。** GNN-only 走步的门（`gnn_only.rs::rank_window`，批次 = `CURRENT` + `c0..cN`）
走的是 serve 的**批量**路径：候选数 >=2 ⇒ `predict_rank_batch` 返回「逐模型竞争名次的平均」
（serve.py:361-403）；而 `evaluate` 的接受判据走的是**单候选**路径（n<2 ⇒ 同函数 :375-378
直接 `np.mean(pm)`，原始预测延迟）。走步那趟在 438/446 个窗口上由前者判「CURRENT 最好」，
整趟只评估了 8 个候选（`evals=9`）。这个脚本把**同一批**分别喂两条路径，再拿 truth.csv 判谁对：

  rank   口径 —— 直接用代理记账下来的 serve 响应（与 Rust 当时看到的一字不差）
  single 口径 —— 每个去重后的电路单发一次 /rank（n=1）拿原始预测延迟
  truth  口径 —— `TL_SIM_DIR` 补出来的真仿真值

三个口径各给一个「CURRENT 是不是最好」的计数，三者一对照就知道门错在哪一环。

**为什么用代理取 netlist。** `.tl` → SPICE 网表只能在 Rust 里转（`module_to_candidate`），
但 `GnnClient` 的目标地址是环境变量给的 ⇒ 把 GNN_PORT 指到一个记账代理上，就能**零改动、
零重编译**拿到 Rust 实际发出的候选 JSON。代理只转发 + 记账，绝不改请求或响应。

用法（服务器；serve 在 127.0.0.1:8000，本脚本系统 python3 即可）：
  proxy : python3 scripts/diag/_t_rank_probe.py proxy --listen 8001 --upstream 8000 --out BODIES.jsonl
  replay: python3 scripts/diag/_t_rank_probe.py replay --bodies BODIES.jsonl --windows windows.csv \\
                 --truth truth.csv --out replay.csv

**对齐（replay 的关键一步，必须自证）。** 代理只拿到 netlist，没有 `mod_*.tl` 文件名，
而 truth.csv 是按文件名索引的。对齐靠**位置**：Rust 的 `order` = scored 按
(名次升序, 下标升序)、再拼上未打分的；capture 的 `candidates_ranked` 正是按 `order` 写的
文件名 ⇒ 「本批响应的名次序」与 `candidates_ranked` 的前 `n_scored` 个逐个对应，
`CURRENT` 对应 `cur_file`。
⚠ 对齐必须自证：拿候选里的 `transistor_count` 与 truth.csv 同名行的 `transistor_count` 比，
不符的窗口在报告里单列 —— 对齐错了下面的结论全是废的。
"""
import argparse
import hashlib
import json
import sys
import threading

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

KEY_FIELDS = ('netlist', 'input_pins', 'output_pins', 'gate_logics', 'transistor_count')


# ── 代理 ────────────────────────────────────────────────────────────────────

def cmd_proxy(a):
    from http.client import HTTPConnection
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    fh = open(a.out, 'a', encoding='utf-8')
    lock = threading.Lock()
    stat = {'n': 0, 'batch': 0, 'single': 0, 'err': 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):     # 不刷屏
            pass

        def _forward(self, method, raw):
            conn = HTTPConnection(a.upstream_host, a.upstream, timeout=a.timeout)
            hdrs = {}
            ct = self.headers.get('Content-Type')
            if ct:
                hdrs['Content-Type'] = ct
            conn.request(method, self.path, body=raw if raw else None, headers=hdrs)
            resp = conn.getresponse()
            data, code = resp.read(), resp.status
            conn.close()
            return data, code

        def _reply(self, data, code):
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                data, code = self._forward('GET', b'')
            except Exception as e:
                self._reply(json.dumps({'error': str(e)}).encode(), 502)
                return
            self._reply(data, code)

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(n) if n else b''
            try:
                data, code = self._forward('POST', raw)
            except Exception as e:
                with lock:
                    stat['err'] += 1
                self._reply(json.dumps({'error': str(e)}).encode(), 502)
                return
            self._reply(data, code)
            try:
                req = json.loads(raw.decode('utf-8'))
                resp = json.loads(data.decode('utf-8'))
                cands = req.get('candidates') or []
                nc = len(cands)
                with lock:
                    stat['n'] += 1
                    stat['batch' if nc >= 2 else 'single'] += 1
                    fh.write(json.dumps({'path': self.path, 'n': nc, 'req': cands,
                                         'resp': resp}, ensure_ascii=False) + '\n')
                    fh.flush()
            except Exception as e:
                with lock:
                    stat['err'] += 1
                    fh.write(json.dumps({'path': self.path, 'error': str(e)}) + '\n')
                    fh.flush()

    srv = ThreadingHTTPServer((a.listen_host, a.listen), Handler)
    print(f'[proxy] 监听 {a.listen_host}:{a.listen} → {a.upstream_host}:{a.upstream}  '
          f'记账 {a.out}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        print(f"[proxy] 收工：请求 {stat['n']}（批量 {stat['batch']} / 单发 {stat['single']}）"
              f"  记账失败 {stat['err']}", flush=True)


# ── 回放 ────────────────────────────────────────────────────────────────────

def cand_key(c):
    payload = json.dumps({k: c.get(k) for k in KEY_FIELDS}, sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def post_rank(host, port, cands, timeout):
    import urllib.request
    body = json.dumps({'candidates': cands}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(f'http://{host}:{port}/rank', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode('utf-8'))


def read_truth(path):
    """truth.csv: file,status,avg_delay,transistor_count,error（error 含逗号 ⇒ splitn 5）。"""
    out = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if not line or line.startswith('file,'):
                continue
            p = line.split(',', 4)
            if len(p) < 4:
                continue
            delay = trans = None
            if p[1] == 'ok' and p[2]:
                try:
                    delay = float(p[2])
                    trans = int(p[3])
                except ValueError:
                    delay = trans = None
            out[p[0]] = (delay, trans)
    return out


def read_windows(path):
    """windows.csv 末列是逗号连接的候选名 ⇒ 整行不能朴素 split(',')，用 split(',', 8)。"""
    rows = []
    with open(path, 'r', encoding='utf-8') as fh:
        fh.readline()
        for line in fh:
            line = line.rstrip('\r\n')
            if not line:
                continue
            r = line.split(',', 8)
            if len(r) < 9:
                continue
            rows.append({'iter': r[0], 'window': r[1], 'eval_idx': r[2], 'any_better': r[3],
                         'n_scored': int(r[4]), 'n_candidates': int(r[5]), 'cur_rank': r[6],
                         'cur_file': r[7], 'cands': [c for c in r[8].split(',') if c]})
    return rows


def kendall_tau(xs, ys):
    """无并列修正的 τ-a；并列多时只当粗指标用（本处名次是半整数，并列确实存在）。"""
    n = len(xs)
    if n < 3:
        return None
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                con += 1
            else:
                dis += 1
    tot = con + dis
    return (con - dis) / tot if tot else None


def cmd_replay(a):
    recs = []
    with open(a.bodies, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get('n', 0) >= 2 and o.get('req') and o.get('resp'):
                recs.append(o)
    if not recs:
        print('[replay] 记账里没有批量请求 —— 代理那趟没开着 GNN_WALK？')
        return
    print(f'[replay] 记账里的窗口批次 = {len(recs)}')

    uniq = {}
    for o in recs:
        for c in o['req']:
            uniq.setdefault(cand_key(c), c)
    print(f'[replay] 去重后的不同电路 = {len(uniq)}（逐个单发一次拿原始预测延迟）')

    keys = sorted(uniq)
    if a.limit:
        keys = keys[:a.limit]
    single = {}
    for i, k in enumerate(keys):
        try:
            single[k] = float(post_rank(a.host, a.port, [uniq[k]], a.timeout)['ranked'][0]['avg_delay'])
        except Exception as e:
            single[k] = None
            print(f'  单发失败 {k}: {e}')
        if (i + 1) % 25 == 0 or i + 1 == len(keys):
            print(f'  单发进度 {i + 1}/{len(keys)}', flush=True)

    wins = read_windows(a.windows)
    print(f'[replay] windows.csv 窗口 = {len(wins)}   批次数 = {len(recs)}'
          + ('' if len(wins) == len(recs) else '   ⚠ 两者不等，按较短的逐行对齐'))
    truth = read_truth(a.truth)
    print(f'[replay] truth.csv 行 = {len(truth)}')

    key2file = {}          # 电路 → mod_*.tl 文件名（靠窗口内位置对齐反推）
    key2win = {}           # 电路 → 出现在多少个窗口里
    rows = []
    for o, w in zip(recs, wins):
        req = o['req']
        rk = {e['id']: e['avg_delay'] for e in o['resp'].get('ranked', [])}
        cur_rank = rk.get('CURRENT')
        if cur_rank is None or cur_rank != cur_rank:
            continue
        # 与 Rust 的 order 同序：有限分按 (名次升序, 下标升序)，未打分的按下标接在后面
        idxs = list(range(len(req) - 1))

        def sk(i):
            v = rk.get(f'c{i}')
            return (0, v, i) if isinstance(v, float) and v == v else (1, 0.0, i)
        idxs.sort(key=sk)
        files = w['cands']
        n_scored = min(w['n_scored'], len(idxs), len(files))
        # 对齐自证：transistor_count 必须与 truth.csv 同名行一致
        for i in range(n_scored):
            tc = req[1 + idxs[i]].get('transistor_count')
            tt = truth.get(files[i], (None, None))[1]
            if tc is not None and tt is not None and int(tc) != int(tt):
                n_scored = i
                break

        cur_c = req[0]
        cur_key = cand_key(cur_c)
        key2file.setdefault(cur_key, w['cur_file'])
        key2win[cur_key] = key2win.get(cur_key, 0) + 1
        cur_single = single.get(cur_key)
        cur_true = truth.get(w['cur_file'], (None, None))[0]
        cand_rank, cand_single, cand_true = [], [], []
        for i in range(n_scored):
            c = req[1 + idxs[i]]
            k = cand_key(c)
            key2file.setdefault(k, files[i])
            key2win[k] = key2win.get(k, 0) + 1
            cand_rank.append(rk.get(f'c{idxs[i]}'))
            cand_single.append(single.get(k))
            cand_true.append(truth.get(files[i], (None, None))[0])
        rows.append({'iter': w['iter'], 'window': w['window'], 'eval_idx': w['eval_idx'],
                     'any_better': w['any_better'], 'n_scored': n_scored,
                     'n_candidates': w['n_candidates'], 'cur_file': w['cur_file'],
                     'cur_rank': cur_rank, 'cur_single': cur_single, 'cur_true': cur_true,
                     'cand_rank': cand_rank, 'cand_single': cand_single, 'cand_true': cand_true})

    # ── 落盘 ────────────────────────────────────────────────────────────────
    with open(a.out, 'w', encoding='utf-8', newline='') as fh:
        fh.write('iter,window,eval_idx,cur_rank,cur_single,cur_true,best_cand_rank,'
                 'best_cand_single,best_cand_true,n_scored,n_candidates,cur_file,cur_is_best_rank,'
                 'cur_is_best_single,cur_is_best_true\n')
        for r in rows:
            cr = [v for v in r['cand_rank'] if isinstance(v, float) and v == v]
            cs = [v for v in r['cand_single'] if isinstance(v, float) and v == v]
            ct = [v for v in r['cand_true'] if v is not None]
            br, bs, bt = (min(cr) if cr else None, min(cs) if cs else None,
                          min(ct) if ct else None)
            fh.write(f"{r['iter']},{r['window']},{r['eval_idx']},{r['cur_rank']:.3f},"
                     f"{'' if r['cur_single'] is None else format(r['cur_single'], '.6e')},"
                     f"{'' if r['cur_true'] is None else format(r['cur_true'], '.6e')},"
                     f"{'' if br is None else format(br, '.3f')},"
                     f"{'' if bs is None else format(bs, '.6e')},"
                     f"{'' if bt is None else format(bt, '.6e')},"
                     f"{r['n_scored']},{r['n_candidates']},{r['cur_file']},"
                     f"{int(br is not None and r['cur_rank'] <= br)},"
                     f"{int(bs is not None and r['cur_single'] is not None and r['cur_single'] <= bs)},"
                     f"{int(bt is not None and r['cur_true'] is not None and r['cur_true'] <= bt)}\n")

    # ── 逐电路标定表 ────────────────────────────────────────────────────────
    # 对齐反推出的 key→文件名，配上单发预测与真值 ⇒ 部署分布上的标定曲线。
    # `n_win` 是同一电路出现在多少个窗口里（191 那种「一个电路被反复比」的规模一目了然）。
    n_cal = 0
    if a.cands_out:
        with open(a.cands_out, 'w', encoding='utf-8', newline='') as fh:
            fh.write('file,single_pred,true_delay,transistor_count,ratio_pred_over_true,n_win,key\n')
            for k, c in uniq.items():
                f = key2file.get(k, '')
                td, tc = truth.get(f, (None, None))
                sp = single.get(k)
                ratio = (f'{sp / td:.6f}' if (sp and td) else '')
                if f and td is not None and sp:
                    n_cal += 1
                fh.write(f'{f},{"" if sp is None else format(sp, ".6e")},'
                         f'{"" if td is None else format(td, ".6e")},'
                         f'{"" if c.get("transistor_count") is None else c["transistor_count"]},'
                         f'{ratio},{key2win.get(k, 0)},{k}\n')
        print(f'[replay] 逐电路标定表 → {a.cands_out}（模型值+真值+管数齐全 {n_cal} 行）')

    # ── 三个口径对照 ────────────────────────────────────────────────────────
    def tally(f, sel):
        n = sum(1 for r in rows if r[sel] is not None)
        k = sum(1 for r in rows if r[sel] is not None and f(r))
        return k, n
    kr, nr = tally(lambda r: r['cur_rank'] <= min([v for v in r['cand_rank']
                                                   if isinstance(v, float) and v == v] or [1e18]),
                   'cur_rank')
    ks, ns = tally(lambda r: r['cur_single'] <= min([v for v in r['cand_single']
                                                     if isinstance(v, float) and v == v] or [1e18]),
                   'cur_single')
    kt, nt = tally(lambda r: r['cur_true'] <= min([v for v in r['cand_true']
                                                   if v is not None] or [1e18]),
                   'cur_true')
    taus = []
    for r in rows:
        xs = [r['cur_rank']] + r['cand_rank']
        ys = [r['cur_single']] + r['cand_single']
        pairs = [(x, y) for x, y in zip(xs, ys)
                 if isinstance(x, float) and x == x and isinstance(y, float) and y == y]
        t = kendall_tau([p[0] for p in pairs], [p[1] for p in pairs])
        if t is not None:
            taus.append(t)
    taus.sort()

    print()
    print('=' * 88)
    print(f'三个口径「CURRENT 是这批最好」的计数（窗口 {len(rows)} 个）')
    print('=' * 88)
    print(f'  rank   口径（走步门用的那个）: {kr}/{nr}')
    print(f'  single 口径（evaluate 判据用的）: {ks}/{ns}')
    print(f'  truth  口径（真仿真）          : {kt}/{nt}')
    if taus:
        print(f'\n逐窗 rank 序 vs single 序的 Kendall τ：中位数 {taus[len(taus) // 2]:+.3f}   '
              f'均值 {sum(taus) / len(taus):+.3f}   （n={len(taus)}）')
    n_any = sum(1 for r in rows if r['any_better'] == '1')
    print(f'走步记录 any_better=1 的窗口 = {n_any}（模型侧认定有候选更强）')
    print(f'\n逐窗明细 → {a.out}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('proxy')
    p.add_argument('--listen', type=int, default=8001)
    p.add_argument('--listen-host', default='127.0.0.1')
    p.add_argument('--upstream', type=int, default=8000)
    p.add_argument('--upstream-host', default='127.0.0.1')
    p.add_argument('--out', required=True)
    p.add_argument('--timeout', type=float, default=1800.0)
    p.set_defaults(func=cmd_proxy)

    r = sub.add_parser('replay')
    r.add_argument('--bodies', required=True)
    r.add_argument('--windows', required=True)
    r.add_argument('--truth', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--cands-out', default='', help='逐电路标定表（模型值 vs 真值 vs 管数）')
    r.add_argument('--host', default='127.0.0.1')
    r.add_argument('--port', type=int, default=8000)
    r.add_argument('--timeout', type=float, default=600.0)
    r.add_argument('--limit', type=int, default=0, help='只单发前 N 个电路（试跑用）')
    r.set_defaults(func=cmd_replay)

    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
