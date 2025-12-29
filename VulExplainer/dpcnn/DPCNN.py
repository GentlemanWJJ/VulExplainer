# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCNN(nn.Module):

    def __init__(
        self, encoder, tokenizer, dim_channel, num_blocks, num_class, args
    ):
        super(DPCNN, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        emb_dim = encoder.config.hidden_size

        # 初始卷积层
        self.initial_conv = nn.Conv2d(1, dim_channel, (3, emb_dim), padding=(1, 0))

        # 卷积块（包含两个卷积层和残差连接）
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(dim_channel, dim_channel, (3, 1), padding=(1, 0)),
                    nn.ReLU(),
                    nn.Conv2d(dim_channel, dim_channel, (3, 1), padding=(1, 0)),
                )
                for _ in range(num_blocks)
            ]
        )

        # 池化层（下采样）
        self.pool = nn.MaxPool2d((3, 1), stride=2, padding=(1, 0))

        # Dropout和全连接层
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(dim_channel, num_class)

    def forward(self, input_ids):
        # 获取词嵌入
        emb_x = self.encoder.embeddings(input_ids)  # [batch_size, seq_len, emb_dim]

        # 注意力掩码
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(-1).expand_as(emb_x)
        )

        emb_x = emb_x * attention_mask  # [batch_size, seq_len, emb_dim]

        # 调整维度用于卷积 [batch_size, 1, seq_len, emb_dim]
        x = emb_x.unsqueeze(1)

        # 初始卷积
        x = self.initial_conv(x)  # [batch_size, dim_channel, seq_len, 1]

        # 金字塔卷积块
        for block in self.conv_blocks:
            residual = x
            x = block(x)  # [batch_size, dim_channel, seq_len, 1]
            x = x + residual  # 残差连接
            x = self.pool(x)  # 下采样 [batch_size, dim_channel, seq_len//2, 1]

        # 全局最大池化
        x = F.max_pool1d(x.squeeze(-1), x.size(2)).squeeze(
            -1
        )  # [batch_size, dim_channel]

        # 输出层
        x = self.dropout(x)
        return x


class DPCNN2(nn.Module):

    def __init__(
        self, encoder, tokenizer, dim_channel, num_blocks, num_class, args
    ):
        super(DPCNN2, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        emb_dim = encoder.config.hidden_size
        self.conv_region = nn.Conv2d(1, dim_channel, (3, emb_dim), stride=1)
        self.conv = nn.Conv2d(dim_channel, dim_channel, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm2d(num_features=dim_channel)
        self.fc = nn.Linear(dim_channel, num_class)

    def forward(self, input_ids):
        x = self.encoder.embeddings(input_ids)
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(-1).expand_as(x)
        )
        x = x * attention_mask  # [batch_size, seq_len, emb_dim]
        x = x.unsqueeze(1)  # [batch_size, dim_channel, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, dim_channel, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, dim_channel, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, dim_channel, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, dim_channel, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, dim_channel, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, dim_channel]
        x= self.dropout(x)
        # x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
