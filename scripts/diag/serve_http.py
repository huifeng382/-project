"""serve_http.py — HTTP 版 GNN 粗筛排序服务（serve.py 跑 orca，Rust 本地调）。

用法（orca）：
  ~/venv/bin/python3 scripts/diag/serve_http.py \
      --ckpt <ckpt1.pt> <ckpt2.pt> ... --scaler outputs/scaler.pkl --port 8000

端点：
  POST /rank
    body: {"candidates": [{"id": "...", "netlist": "...",
                           "input_pins": [...], "output_pins": [...]}, ...]}
    resp: {"ranked": [{"id": "...", "avg_delay": 3.2e-11}, ...]}  # 按 avg_delay 升序

Rust（本地）调用示例：curl -X POST http://<orca-ip>:8000/rank -H 'Content-Type: application/json' -d @candidates.json
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

# 复用 serve.py 的配置/建图/预测逻辑（serve.py 里已对齐 STRUCT_MODE=logic_only 等）
import scripts.diag.serve as S
from src.utils import load_scaler

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS, SCALER, DEVICE = [], None, 'cpu'


class RankHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip('/') != '/rank':
            self._send(404, {'error': 'not found'}); return
        try:
            ln = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(ln) or b'{}')
            cands = data.get('candidates', data) if isinstance(data, dict) else data
            if not isinstance(cands, list):
                cands = []
        except Exception as e:
            self._send(400, {'error': f'bad request: {e}'}); return
        results = []
        for c in cands:
            try:
                ad = S.predict_avg_delay(MODELS, c.get('netlist', ''),
                                         c.get('input_pins', []), c.get('output_pins', []),
                                         SCALER, DEVICE, gate_logics=c.get('gate_logics'))
                results.append({'id': c.get('id', '?'), 'avg_delay': ad})
            except Exception as e:
                results.append({'id': c.get('id', '?'), 'avg_delay': None, 'error': str(e)})
        results.sort(key=lambda r: (r['avg_delay'] is None, r['avg_delay']))
        self._send(200, {'ranked': results})

    def do_GET(self):
        self._send(200, {'status': 'ok', 'n_models': len(MODELS)})

    def _send(self, code, obj):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"[http] {self.client_address[0]} {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--scaler', default=os.path.join('outputs', 'scaler.pkl'))
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()

    global MODELS, SCALER, DEVICE
    SCALER = load_scaler(args.scaler) if os.path.exists(args.scaler) else None
    # 用候选维度需要至少一个样例；这里用一个占位网表确定 in_dim
    # （实际请求时 predict_avg_delay 会按各候选自己的图重建，in_dim 固定由 STRUCT_MODE 决定）
    sample_net = ".SUBCKT DUT a y vdd gnd\nX_1 a wire_1 SC_INV\nX_2 wire_1 y SC_INV\n.ENDS DUT"
    MODELS, in_dim = S.load_models(args.ckpt, SCALER, sample_net, ['a'], ['y'])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for m in MODELS:
        m.to(DEVICE)
    print(f"[serve_http] {len(MODELS)} 模型集成, in_dim={in_dim}, STRUCT_MODE={S.config.STRUCT_MODE}")
    print(f"[serve_http] 监听 {args.host}:{args.port}  POST /rank")
    ThreadingHTTPServer((args.host, args.port), RankHandler).serve_forever()


if __name__ == '__main__':
    main()
