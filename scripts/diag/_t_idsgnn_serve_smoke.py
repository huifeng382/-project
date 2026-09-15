"""_t_idsgnn_serve_smoke.py — serve 模式 '3' 的本地端到端冒烟（用随机权重的假 ckpt，不需要服务器）。

造两折 idsavg GNN 假 ckpt + 一个 in_dim=15 的 delay 假 ckpt，跑 serve.predict_rank_batch 的
mode '3'（GNN 现场列）与 mode '1'（线性近似）两条路径，确认：
  - GNN 折模型能载入、per-(pin,dir) 表能算出、列宽 8 静态 + 7 动态 + 1 ids = 15 对上 delay ckpt
  - mode '1' 旧路径没被改坏（回归）
  - GNN 列**逐行不同**（这正是 '3' 与 '1'/'2' 的关键差别；若逐行相同说明又预算了一次）

用法: python3 scripts/diag/_t_idsgnn_serve_smoke.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'diag'))

TMP = tempfile.mkdtemp(prefix='_smoke_idsgnn_')
os.environ['IDSGNN_CKPT'] = os.path.join(TMP, 'idsgnn_fold*.pt')
os.environ['USE_IDS_AVG_APPROX'] = '3'
os.environ['STRUCT_MODE'] = 'logic_only'

import pandas                           # noqa: F401,E402  ⚠ 必须早于 torch：本机 Windows 上
# "torch 先加载、pandas 后加载" 会触发 PyG→pandas→zoneinfo 的堆损坏 (0xc0000374, exit 127)，
# 与本改动无关（src/model.py 未改，单独 import 亦然）。服务器 Linux 无此问题。
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import config                           # noqa: E402
import src.graph_builder as gb          # noqa: E402

# ---- 造假 idsavg GNN 折 ckpt（元信息按 R4: base 静态 7 + CONE_V2 + CONE_PIN）
from _idsgnn_serve import IdsAvgGNN     # noqa: E402
N_CONT_BASE, CONE_N, PIN_N = 7, 6, 3
N_EXTRA = 7 + CONE_N + PIN_N
N_CONT = N_CONT_BASE + N_EXTRA
STA_MEAN = [0.0] * N_CONT_BASE
STA_STD = [1.0] * N_CONT_BASE
for i in range(2):
    torch.manual_seed(i)
    m = IdsAvgGNN(17, N_CONT, emb=32, hid=160, K=5, dropout=0.25)
    torch.save({'state_dict': m.state_dict(), 'num_types': 17, 'n_cont': N_CONT,
                'emb': 32, 'hid': 160, 'k': 5, 'dropout': 0.25,
                'struct_mode': 'base', 'n_cont_base': N_CONT_BASE, 'cone_v2': True,
                'cone_feat': False, 'cone_pin': True,
                'sta_mean': STA_MEAN, 'sta_std': STA_STD,
                'val_r2': 0.79, 'best_ep': 100},
               os.path.join(TMP, f'idsgnn_fold{i}.pt'))
print(f'[smoke] 假 idsavg GNN ckpt → {TMP}')

import serve as _serve                   # noqa: E402  (环境变量已设，import 时才定 _APPROX_MODE)
assert _serve._APPROX_MODE == '3', _serve._APPROX_MODE

# ---- 假 delay ckpt：in_dim = 静态 7 + 动态 7 + ids 1 = 15（serve 的 in_dim 推导规则）
from src.model import DelayGNN           # noqa: E402
cands = json.load(open(os.path.join(ROOT, 'models', 'candidates_test.json'), encoding='utf-8'))
c0 = cands[0]
_serve.rebuild_gate_types(_serve._load_cell_types_from_netlist(c0['netlist']))
gb.rebuild_gate_types(gb.LOGIC_TYPES + ['INPUT_PIN', 'OUTPUT_PIN', 'UNKNOWN_GATE'])
torch.manual_seed(0)
dm = DelayGNN(in_dim=15, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
              dropout=config.DROPOUT, num_gate_types=len(gb.GATE_TYPES),
              gate_embed_dim=config.GATE_EMBED_DIM)
CK = os.path.join(TMP, 'delay_fake.pt')
torch.save(dm.state_dict(), CK)

models, in_dim = _serve.load_models([CK], None, c0['netlist'], c0['input_pins'], c0['output_pins'])
print(f'[smoke] delay 模型 in_dim={in_dim}（应为 15）')
assert in_dim == 15, in_dim

dev = torch.device('cpu')
for m in models:
    m.to(dev)
out3 = _serve.predict_rank_batch(models, cands[:4], None, dev)
print('[smoke] mode 3 结果:', [(r['id'], round(r['avg_delay'], 4)) for r in out3])
assert all(isinstance(r['avg_delay'], float) and r['avg_delay'] == r['avg_delay'] for r in out3), out3

# ---- 检查：GNN 列是否真的逐行不同（'3' 的立身之本）
srv = _serve._idsgnn_prepare()
nn, nsd, ei, pins, outs, gid = _serve.build_candidate_tensors(
    c0['netlist'], c0['input_pins'], c0['output_pins'], None)
print(f'[smoke] 候选图 N={len(nn)} 静态列={nsd.shape[1]}（logic_only 应为 7）行数={len(gid)}')
assert nsd.shape[1] == 7, nsd.shape[1]
assert set(k[1] for k in gid) == {'rise', 'fall'}, gid.keys()
vecs = {k: v for k, v in gid.items()}
first = list(vecs.values())[0]
assert all(v.shape == first.shape for v in vecs.values())
diff_rows = sum(1 for v in vecs.values() if not np.allclose(v, first))
print(f'[smoke] 与首行不同的行数 = {diff_rows}/{len(vecs)}（>0 才算逐行变化）')
assert diff_rows > 0, 'GNN 列在所有 (pin,dir) 行上完全相同 → 退化成模式 2 式预算，接线有误'
assert np.abs(first).max() > 0, 'GNN 列全 0 → 前向没跑通'

# ---- 回归：mode '1' 线性近似路径仍可用
os.environ['USE_IDS_AVG_APPROX'] = '1'
import importlib
importlib.reload(_serve)
assert _serve._APPROX_MODE == '1'
models1, in_dim1 = _serve.load_models([CK], None, c0['netlist'], c0['input_pins'], c0['output_pins'])
for m in models1:
    m.to(dev)
out1 = _serve.predict_rank_batch(models1, cands[:4], None, dev)
print(f"[smoke] mode 1 结果: {[(r['id'], round(r['avg_delay'], 4)) for r in out1]}")
print(f'[smoke] mode 3 排序 id 序 = {[r["id"] for r in out3]}')
print(f'[smoke] mode 1 排序 id 序 = {[r["id"] for r in out1]}')
assert all(r['avg_delay'] == r['avg_delay'] for r in out1)

print('\n[smoke] OK —— 模式 3 接线端到端跑通，且未破坏模式 1')
