import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

class DelayGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 num_gate_types=100, gate_embed_dim=32):
        super().__init__()
        # 门类型 Embedding：把 1 维整数索引映射为 gate_embed_dim 维稠密向量
        self.gate_embed = nn.Embedding(num_gate_types, gate_embed_dim)
        # 实际输入维度 = embed_dim + (in_dim - 1)  （in_dim 包含 gate_idx + 3 结构 + 5 动态）
        actual_in_dim = gate_embed_dim + (in_dim - 1)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GraphConv(actual_in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # x[:, 0] 是门类型索引，x[:, 1:] 是结构+动态特征
        gate_idx = x[:, 0].long()
        struct_dyn = x[:, 1:]
        gate_emb = self.gate_embed(gate_idx)
        x = torch.cat([gate_emb, struct_dyn], dim=1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.squeeze(-1)