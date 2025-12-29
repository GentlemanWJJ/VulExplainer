"""
使用高级分类器的FusionModel版本
展示如何集成各种现代分类模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils import preprocess_adj, preprocess_features
from DPCNN import DPCNN, DPCNN2
from ReGCN import ReGCN
from TextRCNN import TextRCNN
from advanced_classifiers import (
    MLPClassifier,
    AttentionClassifier,
    ResidualClassifier,
    TransformerClassifier,
    LabelAwareClassifier,
    ContrastiveClassifier,
    HierarchicalClassifier,
    CapsuleClassifier
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedFusionModel(nn.Module):
    """
    使用高级分类器的融合模型
    支持多种现代分类器选择
    """

    def __init__(self, encoder, tokenizer, args, num_class, classifier_type='mlp'):
        super(AdvancedFusionModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.classifier_type = classifier_type

        self.dpcnn_out_dim = args.hidden_size
        self.regcn_out_dim = args.hidden_size 
        self.num_class = num_class

        # 初始化DPCNN组件
        self.dpcnn = DPCNN(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dim_channel=self.dpcnn_out_dim,
            num_blocks=3,
            num_class=num_class,
            args=self.args,
        )

        self.TextRCNN = TextRCNN(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            num_class=num_class,
            args=self.args,
        )

        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        self.regcn = ReGCN(
            feature_dim_size=encoder.config.hidden_size,
            hidden_size=self.regcn_out_dim,
            dropout=0.1,
        )

        # 特征融合维度
        self.fusion_dim = 768

        # 根据类型选择分类器
        self.classifier = self._build_classifier(classifier_type)

    def _build_classifier(self, classifier_type):
        """根据类型构建分类器"""
        if classifier_type == 'linear':
            # 原始简单线性分类器
            return nn.Linear(self.fusion_dim, self.num_class)

        elif classifier_type == 'mlp':
            # 多层感知机分类器（推荐：简单有效）
            return MLPClassifier(
                input_dim=self.fusion_dim,
                hidden_dims=[self.fusion_dim, self.fusion_dim // 2],
                num_classes=self.num_class,
                dropout=0.1,
                activation='gelu'
            )

        elif classifier_type == 'attention':
            # 注意力机制分类器（推荐：自适应特征关注）
            return AttentionClassifier(
                input_dim=self.fusion_dim,
                num_classes=self.num_class,
                num_heads=8,
                dropout=0.1
            )

        elif classifier_type == 'residual':
            # 残差连接分类器（推荐：适合深层网络）
            return ResidualClassifier(
                input_dim=self.fusion_dim,
                hidden_dim=self.fusion_dim,
                num_classes=self.num_class,
                num_layers=3,
                dropout=0.1
            )

        elif classifier_type == 'transformer':
            # Transformer分类器（推荐：强大的特征提取）
            return TransformerClassifier(
                input_dim=self.fusion_dim,
                num_classes=self.num_class,
                num_layers=2,
                num_heads=8,
                hidden_dim=self.fusion_dim,
                dropout=0.1
            )

        elif classifier_type == 'label_aware':
            # 标签感知分类器（推荐：适合多分类任务）
            return LabelAwareClassifier(
                input_dim=self.fusion_dim,
                num_classes=self.num_class,
                label_embed_dim=128,
                dropout=0.1
            )

        elif classifier_type == 'contrastive':
            # 对比学习分类器（推荐：提升特征质量）
            return ContrastiveClassifier(
                input_dim=self.fusion_dim,
                num_classes=self.num_class,
                projection_dim=256,
                dropout=0.1
            )

        elif classifier_type == 'hierarchical':
            # 层次化分类器（推荐：适合CWE等层次结构）
            return HierarchicalClassifier(
                input_dim=self.fusion_dim,
                num_classes=self.num_class,
                num_hierarchies=2,
                dropout=0.1
            )

        elif classifier_type == 'capsule':
            # 胶囊网络分类器（推荐：捕获空间关系）
            return CapsuleClassifier(
                input_dim=self.fusion_dim,
                num_classes=self.num_class,
                num_capsules=10,
                capsule_dim=16,
                dropout=0.1
            )

        else:
            raise ValueError(f"未知的分类器类型: {classifier_type}")

    def forward(self, input_ids, labels):
        # 获取DPCNN的特征
        # dpcnn_features = self.dpcnn(input_ids)  # [batch_size, dim_channel]

        # 可以在这里添加其他特征融合
        # rcnn_features = self.TextRCNN(input_ids)
        # regcn_features = ...
        outputs = self.encoder(
            input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        ).last_hidden_state
        outputs = outputs[:, 0, :]
        fused_features = outputs

        # 分类预测
        if self.classifier_type == 'contrastive':
            logits = self.classifier(fused_features, return_projection=False)
        else:
            logits = self.classifier(fused_features)  # [batch_size, num_class]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return F.softmax(logits, dim=-1)


# 为了向后兼容，保留原始FusionModel
class FusionModel(nn.Module):
    """原始FusionModel，使用简单线性分类器"""
    
    def __init__(self, encoder, tokenizer, args, num_class):
        super(FusionModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args

        self.dpcnn_out_dim = args.hidden_size
        self.regcn_out_dim = args.hidden_size 
        self.num_class = num_class
        
        self.dpcnn = DPCNN(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dim_channel=self.dpcnn_out_dim,
            num_blocks=3,
            num_class=num_class,
            args=self.args,
        )
        
        self.TextRCNN = TextRCNN(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            num_class=num_class,
            args=self.args,
        )
        
        self.w_embeddings = self.encoder.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        self.regcn = ReGCN(
            feature_dim_size=encoder.config.hidden_size,
            hidden_size=self.regcn_out_dim,
            dropout=0.1,
        )
        
        self.fusion_dim = self.dpcnn_out_dim
        self.classifier = nn.Linear(self.fusion_dim, self.num_class)

    def forward(self, input_ids, labels):
        dpcnn_features = self.dpcnn(input_ids)
        fused_features = dpcnn_features
        logits = self.classifier(fused_features)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return F.softmax(logits, dim=-1)


weighted_graph = False
print("using default unweighted graph")

def build_graph(shuffle_doc_words_list, word_embeddings, window_size=3):
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
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.0
                    else:
                        word_pair_count[word_pair_key] = 1.0
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
