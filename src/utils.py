# utils.py
import os
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_by_circuit(circuit_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按电路划分训练/验证/测试集，保持电路完整性"""
    set_seed(seed)
    ids = list(circuit_ids)
    random.shuffle(ids)
    n = len(ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    return train_ids, val_ids, test_ids


def _expr_of(cid):
    """从 circuit_id (candidate_expr{N}_{idx}) 提取 expr 组键 (candidate_expr{N})。"""
    s = str(cid)
    return s.rsplit('_', 1)[0] if '_' in s else s


def split_by_expr(circuit_ids, id_to_expr=None, train_ratio=0.7, val_ratio=0.15,
                  test_ratio=0.15, seed=42):
    """按 expr 分组划分：同一 expr 的所有等价变体候选整组进同一 split。
    这是下游「等价变体择优」任务的正确切分——保证 test 有完整变体组、且无 expr 级泄漏。
    先 sorted 再 shuffle，切分与 parquet 行序无关（顺序稳定）。
    id_to_expr: dict[circuit_id -> expr]；缺省则从 circuit_id 解析。"""
    set_seed(seed)
    if id_to_expr is None:
        id_to_expr = {c: _expr_of(c) for c in circuit_ids}
    exprs = sorted(set(id_to_expr.get(c, _expr_of(c)) for c in circuit_ids))
    random.shuffle(exprs)
    n = len(exprs)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))
    train_e = set(exprs[:tr_end]); val_e = set(exprs[tr_end:va_end]); test_e = set(exprs[va_end:])
    def _pick(eset):
        return [c for c in circuit_ids if id_to_expr.get(c, _expr_of(c)) in eset]
    return _pick(train_e), _pick(val_e), _pick(test_e)


def _spearman(pred, true):
    """秩相关（无 scipy 依赖）：ranks 的 Pearson 相关。"""
    n = len(pred)
    if n < 2:
        return np.nan
    rp = np.argsort(np.argsort(pred)).astype(float)
    rt = np.argsort(np.argsort(true)).astype(float)
    if rp.std() == 0 or rt.std() == 0:
        return np.nan
    return float(np.corrcoef(rp, rt)[0, 1])


def ranking_metrics(test_dyn, preds, targets):
    """等价变体择优任务的排序评估。
    按 (expr, corner) 分组，每个变体取「最坏情况延迟」= max over (pin/dir/vector)。
    组内(≥2 变体)算：Spearman、选择遗憾、top-1、捕获率、变体延迟差；
    并做「成对分辨准确率(按真实相对差分档)」——贪心细粒度重写成败的关键。"""
    import pandas as pd
    df = test_dyn.copy().reset_index(drop=True)
    df['_pred'] = np.asarray(preds)
    df['_true'] = np.asarray(targets)
    df['_expr'] = df['expr'].astype(str) if 'expr' in df.columns \
        else df['circuit_id'].astype(str).map(_expr_of)
    if 'corner' not in df.columns:
        df['corner'] = 'x'
    wc = df.groupby(['_expr', 'corner', 'circuit_id']).agg(
        pred=('_pred', 'max'), true=('_true', 'max')).reset_index()
    sps, regrets, top1s, spreads, captured = [], [], [], [], []
    bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, np.inf)]
    labels = ['<2%', '2-5%', '5-10%', '>10%']
    pair_ok = [0] * len(bins); pair_n = [0] * len(bins)
    for (_e, _c), grp in wc.groupby(['_expr', 'corner']):
        if len(grp) < 2:
            continue
        pr = grp['pred'].values; tr = grp['true'].values
        sp = _spearman(pr, tr)
        if not np.isnan(sp):
            sps.append(sp)
        pick = int(np.argmin(pr))           # 模型选的最快变体
        best = int(np.argmin(tr))           # 真正最快
        worst = int(np.argmax(tr))          # 真正最慢
        top1s.append(1.0 if pick == best else 0.0)
        regrets.append((tr[pick] - tr[best]) / max(tr[best], 1e-15) * 100)
        rng = tr[worst] - tr[best]
        if rng > 1e-18:
            spreads.append(rng / max(tr[best], 1e-15) * 100)      # 变体延迟差(相对最优)
            captured.append((tr[worst] - tr[pick]) / rng * 100)   # 捕获率: 抓住最差→最优差距的%
        # 成对分辨（按真实相对差分档）
        m = len(tr)
        for i in range(m):
            for j in range(i + 1, m):
                if abs(tr[i] - tr[j]) < 1e-18:
                    continue
                d = abs(tr[i] - tr[j]) / min(tr[i], tr[j])
                correct = (pr[i] < pr[j]) == (tr[i] < tr[j])
                for bi, (lo, hi) in enumerate(bins):
                    if lo <= d < hi:
                        pair_n[bi] += 1
                        if correct:
                            pair_ok[bi] += 1
                        break
    pair_acc = {labels[bi]: (pair_ok[bi] / pair_n[bi] * 100 if pair_n[bi] else float('nan'),
                             pair_n[bi]) for bi in range(len(bins))}
    return {
        'n_groups': len(top1s),
        'spearman': float(np.mean(sps)) if sps else float('nan'),
        'regret_pct': float(np.mean(regrets)) if regrets else float('nan'),
        'top1_acc': float(np.mean(top1s)) if top1s else float('nan'),
        'spread_pct': float(np.median(spreads)) if spreads else float('nan'),
        'captured_pct': float(np.mean(captured)) if captured else float('nan'),
        'pair_acc': pair_acc,
    }


class GroupedBatchSampler:
    """把同组样本打包进同一 batch（保证 batch 内有变体对，供成对排序损失）。
    group_ids: 长度=数据集(或子集)大小，group_ids[i]=第 i 个样本的组ID。
    每次 __iter__ 用全局 random(已被 set_seed 播种) 重新洗牌 → 每 epoch 不同且可复现。"""
    def __init__(self, group_ids, batch_size, shuffle=True):
        from collections import defaultdict
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        groups = defaultdict(list)
        for i, g in enumerate(group_ids):
            groups[int(g)].append(i)
        self.groups = list(groups.values())
        self._n = len(group_ids)

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(order)
        batch = []
        for gi in order:
            members = self.groups[gi]
            if len(members) > self.batch_size:
                if batch:
                    yield batch; batch = []
                yield members
                continue
            if batch and len(batch) + len(members) > self.batch_size:
                yield batch; batch = []
            batch.extend(members)
        if batch:
            yield batch

    def __len__(self):
        import math
        return max(1, math.ceil(self._n / self.batch_size))


def save_scaler(scaler, path):
    joblib.dump(scaler, path)
def load_scaler(path):
    return joblib.load(path)

def create_dir(path):
    os.makedirs(path, exist_ok=True)