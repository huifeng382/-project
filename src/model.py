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
        # 实际输入维度 = embed_dim + (in_dim - 1)  （in_dim 包含 gate_idx + 3 结构 + N 动态）
        actual_in_dim = gate_embed_dim + (in_dim - 1)

        # GraphConv 图卷积层（对小型电路图更稳定，GAT 对小图容易过拟合）
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers

        # 第一层：输入 → hidden_dim
        self.convs.append(GraphConv(actual_in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 中间层：hidden_dim → hidden_dim（带残差）
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 注意力读出层：让模型学会聚焦于关键节点（如切换引脚），替代 mean_pool
        self.readout_attn = nn.Linear(hidden_dim, 1)

        # 最终预测层
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # x[:, 0] 是门类型索引，x[:, 1:] 是结构+动态特征
        gate_idx = x[:, 0].long()
        struct_dyn = x[:, 1:]
        gate_emb = self.gate_embed(gate_idx)
        x = torch.cat([gate_emb, struct_dyn], dim=1)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 残差连接（第一层维度可能不同，跳过）
            if i > 0 and residual.shape == x.shape:
                x = x + residual

        # 注意力读出：模型学习每个节点的重要性权重
        attn_scores = self.readout_attn(x)  # (N, 1)
        attn_weights = softmax(attn_scores, batch)  # softmax within each graph → sum=1
        x_pooled = global_add_pool(attn_weights * x, batch)  # 加权求和

        x = self.lin(x_pooled)
        return x.squeeze(-1)
