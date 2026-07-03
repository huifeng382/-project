import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_add_pool
from torch_geometric.utils import softmax


class DelayGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 num_gate_types=100, gate_embed_dim=32, gat_heads=4):
        super().__init__()
        # 门类型 Embedding
        self.gate_embed = nn.Embedding(num_gate_types, gate_embed_dim)
        # 实际输入维度 = embed_dim + (in_dim - 1)
        actual_in_dim = gate_embed_dim + (in_dim - 1)

        # GraphConv 图卷积层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers

        # Corner 条件调制：把 corner 的 S/L 值注入每一层，让 GNN 带 corner 感知
        # struct_dyn 中 corner_slew_cond 和 corner_load_cond 的位置（见 data_loader）
        # struct_dyn = [fanout, depth, drive, logic, is_sw, slew, load, out_load,
        #               arrival, corner_slew(9), corner_load(10), gate_state(11)]
        self.corner_mlps = nn.ModuleList()

        # 第一层
        self.convs.append(GraphConv(actual_in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.corner_mlps.append(nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        ))

        # 中间层
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.corner_mlps.append(nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),
            ))

        # 注意力读出层
        self.readout_attn = nn.Linear(hidden_dim, 1)

        # Corner 条件也注入读出层
        self.readout_corner_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 最终预测层
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # x[:, 0] 是门类型索引，x[:, 1:] 是结构+动态特征
        gate_idx = x[:, 0].long()
        struct_dyn = x[:, 1:]
        gate_emb = self.gate_embed(gate_idx)
        x = torch.cat([gate_emb, struct_dyn], dim=1)

        # 提取 corner 条件（每层共用）
        corner_feat = struct_dyn[:, 9:11]  # (N, 2): slew_cond, load_cond

        for i, (conv, norm, corner_mlp) in enumerate(
                zip(self.convs, self.norms, self.corner_mlps)):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            # Corner 条件调制：scale/shift 由 corner S/L 决定
            mod = corner_mlp(corner_feat)        # (N, 2 * hidden_dim)
            scale, shift = mod.chunk(2, dim=-1)   # (N, hidden_dim) each
            x = scale * x + shift
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0 and residual.shape == x.shape:
                x = x + residual

        # 注意力读出
        attn_scores = self.readout_attn(x)
        attn_weights = softmax(attn_scores, batch)
        x_pooled = global_add_pool(attn_weights * x, batch)

        # Corner 条件注入读出：每个图的 corner 特征（同一图内各节点相同）
        corner_pooled = global_add_pool(corner_feat, batch)
        counts = global_add_pool(torch.ones_like(corner_feat[:, :1]), batch)
        corner_per_graph = corner_pooled / counts.clamp(min=1)
        x_pooled = x_pooled + self.readout_corner_mlp(corner_per_graph)

        x = self.lin(x_pooled)
        return x.squeeze(-1)
