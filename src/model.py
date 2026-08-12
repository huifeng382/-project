import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_add_pool
import config


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

        # Corner 条件编码
        self.corner_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # 电路签名编码
        self.sig_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        # 最终预测层（pooled + corner + circuit_sig）
        self.lin = nn.Linear(hidden_dim * 3, 1)
        # per-node 头（LIB 模式用）：从节点特征预测 (slew, load, spare)，softplus 保正
        self.node_pred = nn.Linear(hidden_dim, 3)
        # 结构先验编码器（transistor_count + SC_AND/SC_INV_WIRE 计数 → hidden_dim 残差）
        self.struct_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        # Corner 感知注意力：corner_cond 调制每个节点的池化权重
        self.corner_attn = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch, corner_cond=None, circuit_sig=None, struct_prior=None):
        gate_idx = x[:, 0].long()
        struct_dyn = x[:, 1:]
        gate_emb = self.gate_embed(gate_idx)
        x = torch.cat([gate_emb, struct_dyn], dim=1)

        # 保存 gate_state 掩码（GNN 前），用于路径累加读出
        gate_mask = x[:, -1].clone()  # (N,)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i > 0 and residual.shape == x.shape:
                x = x + residual

        # per-node 头（LIB 模式用）：在路径掩码清零前，从节点特征预测 (slew, load, spare)
        node_sl = F.softplus(self.node_pred(x))  # (N, 3)

        # 路径累加读出：先用 gate_mask 清零非路径节点
        x = gate_mask.unsqueeze(-1) * x  # (N, 1) * (N, H)
        # Corner 注意力：corner_cond 调制每个节点在 pooling 中的权重
        if corner_cond is not None and config.USE_CORNER_ATTN:
            attn_input = torch.cat([x, corner_cond[batch]], dim=-1)  # (N, H+2)
            attn_w = torch.sigmoid(self.corner_attn(attn_input))      # (N, 1)
            x = x * attn_w
        x_pooled = global_add_pool(x, batch)  # (B, H)

        # 注入 corner 条件
        if corner_cond is not None:
            corner_emb = self.corner_encoder(corner_cond)
        else:
            corner_emb = torch.zeros(x_pooled.shape[0], x_pooled.shape[1], device=x_pooled.device)

        # 注入电路签名
        if circuit_sig is not None:
            sig_emb = self.sig_encoder(circuit_sig)
        else:
            sig_emb = torch.zeros(x_pooled.shape[0], x_pooled.shape[1], device=x_pooled.device)

        # 结构先验残差（transistor_count + 门类型计数 → 加 bias）
        if struct_prior is not None:
            x_pooled = x_pooled + self.struct_encoder(struct_prior)

        x_pooled = torch.cat([x_pooled, corner_emb, sig_emb], dim=-1)
        x = self.lin(x_pooled)
        return x.squeeze(-1), node_sl
