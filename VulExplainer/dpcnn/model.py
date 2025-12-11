import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils import preprocess_adj, preprocess_features  # 假设utils中包含这些函数
from DPCNN import DPCNN
from ReGCN import ReGCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args, input_size,num_classes):
        super().__init__()
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(args.hidden_size, num_classes)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FusionModel(nn.Module):

    def __init__(self, encoder, tokenizer, args, num_class):
        super(FusionModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        # Transformer分支默认开启，可通过args.use_transformer控制
        if not hasattr(self.args, "use_transformer"):
            self.args.use_transformer = True
        self.dpcnn_out_dim = args.hidden_size  # DPCNN的通道维度
        self.regcn_out_dim = args.hidden_size 
        self.num_class = num_class
        # 初始化DPCNN组件
        self.dpcnn = DPCNN(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dim_channel=self.dpcnn_out_dim,
            num_blocks=3,
            dropout_rate=0.1,
            num_class=num_class,
            args=self.args,
        )
        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        self.regcn = ReGCN(
            feature_dim_size=encoder.config.hidden_size,
            hidden_size=self.regcn_out_dim,
            num_GNN_layers=args.num_GNN_layers,
            dropout=0.1,
        )
        # 特征融合分类头
        self.fusion_dim = self.dpcnn_out_dim + self.regcn_out_dim
        self.fusion_dim = self.dpcnn_out_dim
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.fusion_dim, self.fusion_dim),
        #     nn.ReLU(),
        #     nn.Dropout(args.dropout_rate),
        #     nn.Linear(self.fusion_dim, args.num_class),
        # )
        # self.classifier = PredictionClassification(args, input_size=self.regcn_out_dim,num_classes=self.num_class)
        self.classifier = nn.Linear(self.fusion_dim, self.num_class)

    def forward(self, input_ids,labels):
        # 获取DPCNN的特征（使用return_hidden_state参数获取池化后的特征）
        dpcnn_features = self.dpcnn(input_ids)  # [batch_size, dim_channel]

        # # 获取ReGCN的特征（通过修改GNNReGVD的forward方法支持返回特征）
        # adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=self.args.window_size)
        # adj, adj_mask = preprocess_adj(adj)
        # adj_feature = preprocess_features(x_feature)
        # adj = torch.from_numpy(adj)
        # adj_mask = torch.from_numpy(adj_mask)
        # adj_feature = torch.from_numpy(adj_feature)
        # regcn_features = self.regcn(
        #     adj_feature.to(device).double(),
        #     adj.to(device).double(),
        #     adj_mask.to(device).double(),
        # )  # [batch_size, regcn_out_dim]

        # 特征融合（拼接）
        # fused_features = torch.cat(
        #     [dpcnn_features, regcn_features], dim=1
        # ).float()  # [batch_size, fusion_dim]
        fused_features = dpcnn_features
        # 分类预测
        logits = self.classifier(fused_features)  # [batch_size, num_class]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return F.softmax(logits, dim=-1)

weighted_graph = False
print("using default unweighted graph")
# build graph function
def build_graph(shuffle_doc_words_list, word_embeddings, window_size=3):
    # print('using window size = ', window_size)
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j : j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p_id = window[p]
                    word_q_id = window[q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.0
                    else:
                        word_pair_count[word_pair_key] = 1.0
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.0
                    else:
                        word_pair_count[word_pair_key] = 1.0

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[p])
            col.append(doc_word_id_map[q])
            weight.append(word_pair_count[key] if weighted_graph else 1.0)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        x_adj.append(adj)

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k])
        x_feature.append(features)

    return x_adj, x_feature
