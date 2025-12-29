import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils import preprocess_adj, preprocess_features  # 假设utils中包含这些函数
from DPCNN import DPCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FusionModel(nn.Module):

    def __init__(self, encoder, tokenizer, args, num_class):
        super(FusionModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args

        self.dpcnn_out_dim = args.hidden_size  # DPCNN的通道维度
        self.num_class = num_class
        # 初始化DPCNN组件
        self.dpcnn = DPCNN(
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            dim_channel=self.dpcnn_out_dim,
            num_blocks=8,
            num_class=num_class,
            args=self.args,
        )
        self.classifier = nn.Linear(self.dpcnn_out_dim, self.num_class)

    def forward(self, input_ids,labels):
        # 获取DPCNN的特征（使用return_hidden_state参数获取池化后的特征）
        dpcnn_features = self.dpcnn(input_ids)  # [batch_size, dim_channel]

        # 分类预测
        logits = self.classifier(dpcnn_features)  # [batch_size, num_class]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return F.softmax(logits, dim=-1)
