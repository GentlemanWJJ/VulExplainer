# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCNN(nn.Module):

    def __init__(
        self, encoder, tokenizer, dim_channel, num_blocks, dropout_rate, num_class, args
    ):
        super(DPCNN, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        emb_dim = encoder.config.hidden_size
        # Transformer编码层超参，默认开启以增强序列上下文建模
        self.use_transformer = getattr(args, "use_transformer", True)
        if self.use_transformer:
            transformer_nhead = getattr(args, "transformer_nhead", 8)
            transformer_ffn_dim = getattr(args, "transformer_ffn_dim", emb_dim * 4)
            transformer_num_layers = getattr(args, "transformer_num_layers", 2)
            transformer_dropout = getattr(args, "transformer_dropout", 0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=transformer_nhead,
                dim_feedforward=transformer_ffn_dim,
                dropout=transformer_dropout,
                batch_first=False,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )
        else:
            self.transformer = None

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
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(dim_channel, num_class)

    def forward(self, input_ids):
        # 获取词嵌入
        emb_x = self.encoder.embeddings(input_ids)  # [batch_size, seq_len, emb_dim]

        # 注意力掩码
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(-1).expand_as(emb_x)
        )

        # 可选的Transformer编码，提升上下文依赖建模
        if self.use_transformer and self.transformer is not None:
            # Transformer使用batch第二维，需转置
            key_padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
            transformer_inp = emb_x.transpose(0, 1)  # [seq_len, batch, emb_dim]
            transformer_out = self.transformer(
                transformer_inp, src_key_padding_mask=key_padding_mask
            )
            emb_x = transformer_out.transpose(0, 1)

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
