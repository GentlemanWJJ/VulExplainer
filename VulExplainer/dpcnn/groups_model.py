# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupModel(nn.Module):

    def __init__(self, encoder, tokenizer, args, num_class):
        super(GroupModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.emb_dim = encoder.config.hidden_size
        self.dim_channel = 256

        self.conv_region = nn.Conv2d(1, self.dim_channel, (3, self.emb_dim), stride=1)
        self.conv = nn.Conv2d(self.dim_channel, self.dim_channel, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.dim_channel, num_class)
        self.dropout = nn.Dropout(0.1)
        self.bn_1 = nn.BatchNorm2d(num_features=self.dim_channel)
        self.bn_2 = nn.BatchNorm2d(num_features=self.dim_channel)

    def _forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.dropout(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = nn.functional.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = nn.functional.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

    def forward(self, input_ids, groups, labels, fgsm_attack=False, lambda_fgsm=0.1):

        emb_x = self.encoder.embeddings(input_ids)
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id)
            .unsqueeze(-1)
            .expand(input_ids.shape[0], input_ids.shape[1], 768)
        )
        embeddings = emb_x * attention_mask

        # 正常前向（干净样本）
        features = self._forward(embeddings)
        logits = self.fc(features)

        if groups is None or not fgsm_attack:
            if groups is None:
                return F.softmax(logits, dim=-1)
            # 仅使用干净样本的交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, groups)
            return loss
