# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextRCNN(nn.Module):

    def __init__(
        self,
        encoder,
        tokenizer,
        num_class,
        args,
    ):
        super(TextRCNN, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.emb_dim = encoder.config.hidden_size
        self.hidden_size = 256
        self.num_layers=2
        self.dropout=0.1
        self.pad_size=32
        self.lstm = nn.LSTM(
            self.emb_dim,
            self.hidden_size,
            self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        # 使用AdaptiveMaxPool1d确保输出维度固定
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        # 添加投影层，将输出维度降到args.hidden_size以匹配classifier
        self.projection = nn.Linear(self.hidden_size * 2 + self.emb_dim, args.hidden_size)
        self.fc = nn.Linear(self.hidden_size * 2 + self.emb_dim, num_class)

    def forward(self, input_ids):
        emb_x = self.encoder.embeddings(input_ids)  # [batch_size, seq_len, emb_dim]

        # 注意力掩码
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(-1).expand_as(emb_x)
        )

        emb_x = emb_x * attention_mask  # [batch_size, seq_len, emb_dim]
        out, _ = self.lstm(emb_x)
        out = torch.cat((emb_x, out), 2)  # [batch_size, seq_len, hidden_size * 2 + emb_dim]
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # [batch_size, hidden_size * 2 + emb_dim, seq_len]
        out = self.maxpool(out)  # [batch_size, hidden_size * 2 + emb_dim, 1]
        out = out.squeeze(-1)  # [batch_size, hidden_size * 2 + emb_dim]
        # 投影到args.hidden_size维度以匹配classifier
        out = self.projection(out)  # [batch_size, args.hidden_size]
        # out = self.fc(out)
        return out
