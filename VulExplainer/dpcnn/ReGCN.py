import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from utils import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

att_op_dict = {"sum": "sum", "mul": "mul", "concat": "concat"}
class ReGCN(nn.Module):
    def __init__(
        self,
        feature_dim_size,
        hidden_size,
        dropout,
        act=nn.functional.relu,
        residual=True,
        att_op="mul",
    ):
        super(ReGCN, self).__init__()
        self.num_GNN_layers = 2
        self.residual = residual
        self.att_op = att_op
        self.out_dim = hidden_size
        if self.att_op == att_op_dict["concat"]:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(
                    GraphConvolution(feature_dim_size, hidden_size, dropout, act=act)
                )
            else:
                self.gnnlayers.append(
                    GraphConvolution(hidden_size, hidden_size, dropout, act=act)
                )
        self.gat_layer = GraphAttentionLayer(
            hidden_size, hidden_size, dropout=dropout, act=act
        )
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else:
                if self.residual:
                    x = (
                        x + self.gnnlayers[idx_layer](x, adj) * mask
                    )  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnnlayers[idx_layer](x, adj) * mask
        x = self.gat_layer(x.double(), adj.double(), mask.double()) * mask.double()
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.att_op == att_op_dict["sum"]:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict["concat"]:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.double(), self.weight.double())
        output = torch.matmul(adj.double(), support.double())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


# 简单的GAT层，支持批量图，使用密集邻接矩阵和padding mask
class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
        alpha=0.2,
        act=torch.relu,
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.act = act

        self.W = Parameter(torch.empty(in_features, out_features))
        self.a = Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj, mask):
        # h: [B, N, F], adj: [B, N, N], mask: [B, N, 1]
        B, N, _ = h.size()
        Wh = torch.matmul(h, self.W.double())  # [B, N, out]

        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leakyrelu(
            torch.matmul(torch.cat([Wh_i, Wh_j], dim=3), self.a.double()).squeeze(3)
        )  # [B, N, N]

        node_mask = mask.squeeze(-1)  # [B, N]
        edge_mask = (adj > 0).float() * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        attention = torch.where(edge_mask > 0, e, torch.full_like(e, -9e15))
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # [B, N, out]
        return self.act(h_prime) if self.act is not None else h_prime
