# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPCNN(nn.Module):

    def __init__(
        self, encoder, tokenizer, args, num_class
    ):
        super(DPCNN, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.emb_dim = encoder.config.hidden_size
        self.dim_channel=256
        self.epsilon = 1e-4  # FGSM扰动系数
        self.tau = getattr(args, "contrastive_tau", 0.1)  # 对比学习温度系数τ

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
        x = x.unsqueeze(1) 
        x = self.conv_region(x) 

        x = self.padding1(x)  # [batch_size, dim_channel, seq_len, 1]
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, dim_channel, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, dim_channel, seq_len, 1]
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, dim_channel, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self.block(x)
        x = x.squeeze()  # [batch_size, dim_channel]
        x = self.dropout(x)
        return x

    def block(self, x):
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

    def InfoNCE(self, clean_feat, adv_feat):
        """
        InfoNCE 对比损失：
        - 正样本对: (clean_feat_i, adv_feat_i)
        - 负样本: 其余样本的 clean / adv 特征
        """
        # clean_feat, adv_feat: [batch_size, hidden_dim]
        batch_size = clean_feat.size(0)
        if batch_size <= 1:
            # batch_size=1 时无法构造负样本，直接返回 0
            return clean_feat.new_tensor(0.0)

        # L2 归一化
        clean_norm = F.normalize(clean_feat, p=2, dim=1)
        adv_norm = F.normalize(adv_feat, p=2, dim=1)

        # 拼接得到 2B 个表示
        reps = torch.cat([clean_norm, adv_norm], dim=0)  # [2B, dim]

        # 相似度矩阵 [2B, 2B]
        sim_matrix = torch.matmul(reps, reps.t()) / self.tau

        # 自身相似度不参与 softmax
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=reps.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # 每一行的正样本索引：
        # 前 B 行的正样本是对应的 adv（索引 +B）
        # 后 B 行的正样本是对应的 clean（索引 -B）
        pos_indices = torch.arange(2 * batch_size, device=reps.device)
        pos_indices[:batch_size] += batch_size
        pos_indices[batch_size:] -= batch_size

        loss_ctr = F.cross_entropy(sim_matrix, pos_indices)
        return loss_ctr

    def forward(self, input_ids,groups, labels, fgsm_attack=True, lambda_fgsm=0.1):
        """
        支持基于 FGSM 的对抗 + 对比学习训练。
        - 生成干净样本与对抗样本的表示
        - 计算交叉熵损失 (干净 + 对抗)
        - 计算 InfoNCE 对比损失
        - 最终总损失：
            L = (1-λ)/2 * (L_CE^v + L_CE^{v+r}) + λ * L_ctr
          其中 λ 由参数 lambda_fgsm 控制，epsilon 为 FGSM 扰动步长。
        """


        # 获取原始embedding
        # embeddings = self.encoder.embeddings(input_ids).detach()
        # embeddings.requires_grad_(True)

        # attention_mask = (
        #     input_ids.ne(self.tokenizer.pad_token_id).unsqueeze(-1).expand_as(embeddings)
        # )
        # embeddings = embeddings * attention_mask  # mask掉pad token

        emb_x = self.encoder.embeddings(input_ids)
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id)
            .unsqueeze(-1)
            .expand(input_ids.shape[0], input_ids.shape[1], 768)
        )
        embeddings = emb_x * attention_mask
        features = self._forward(embeddings)
        logits = self.fc(features)

        if labels is None or not fgsm_attack:
            if labels is None:
                return F.softmax(logits, dim=-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss

        # 训练阶段 + 启用 FGSM 对抗与对比学习
        loss_fct = nn.CrossEntropyLoss()
        loss_clean = loss_fct(logits, labels)

        # 反向传播获取 embedding 梯度
        grads = torch.autograd.grad(
            loss_clean, embeddings, retain_graph=True, create_graph=False
        )[0]

        # FGSM 扰动：按 L2 归一化梯度
        grads_norm = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8
        adv_embeddings = embeddings + self.epsilon * grads / grads_norm

        # 对抗样本前向
        adv_features = self._forward(adv_embeddings)
        adv_logits = self.fc(adv_features)
        loss_adv = loss_fct(adv_logits, labels)

        # InfoNCE 对比损失（干净特征 vs 对抗特征）
        loss_ctr = self.InfoNCE(features, adv_features)

        # 总损失：图片中的公式
        total_loss = (1 - lambda_fgsm) * 0.5 * (loss_clean + loss_adv) + lambda_fgsm * loss_ctr
        return total_loss

