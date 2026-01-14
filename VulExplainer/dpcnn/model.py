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
        self.alpha = getattr(args, "loss_alpha", 0.9)  
        self.conv_region = nn.Conv2d(1, self.dim_channel, (3, self.emb_dim), stride=1)
        self.conv = nn.Conv2d(self.dim_channel, self.dim_channel, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc_labels = nn.Linear(self.dim_channel, num_class)
        self.fc_groups = nn.Linear(self.dim_channel, 6)
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

    def InfoNCE(self, clean_feat, adv_feat, group_ids=None):
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
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=reps.device)
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))
        # ---------- 情况一：无 group_ids，退化为原始 InfoNCE ----------
        if group_ids is None:
            # 每一行的正样本索引：
            # 前 B 行的正样本是对应的 adv（索引 +B）
            # 后 B 行的正样本是对应的 clean（索引 -B）
            pos_indices = torch.arange(2 * batch_size, device=reps.device)
            pos_indices[:batch_size] += batch_size
            pos_indices[batch_size:] -= batch_size
            loss_ctr = F.cross_entropy(sim_matrix, pos_indices)
            return loss_ctr

        # ---------- 情况二：使用粗标签的 Supervised Contrastive ----------
        # group_ids: [batch_size] 或 one-hot [batch_size, num_groups]
        if group_ids.dim() > 1:
            group_ids = group_ids.argmax(-1)

        group_ids = group_ids.view(-1, 1)  # [B, 1]
        labels = torch.cat([group_ids, group_ids], dim=0)  # [2B, 1]

        # 构造正样本掩码：同一粗类别且非自身
        label_mask = torch.eq(labels, labels.T).to(sim_matrix.device)  # [2B, 2B]
        positives_mask = label_mask & (~self_mask)

        # 为了数值稳定，减去每行的最大值
        logits = sim_matrix
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 计算分母：对所有非自身样本求 exp 并求和
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # 只在正样本位置求平均 log_prob
        positives_count = positives_mask.sum(dim=1)
        # 避免除 0：没有正样本的 anchor 其损失视为 0
        positives_count = positives_count.clamp(min=1)
        mean_log_prob_pos = (positives_mask.float() * log_prob).sum(
            dim=1
        ) / positives_count

        loss_ctr = -mean_log_prob_pos.mean()
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

        embeddings = self.encoder.embeddings(input_ids)
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id)
            .unsqueeze(-1)
            .expand_as(embeddings)
        )
        embeddings = embeddings * attention_mask
        features = self._forward(embeddings)
        logits = self.fc_labels(features)
        group_logits = self.fc_groups(features)
        if labels is None:
            return F.softmax(logits, dim=-1)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        groups_loss = loss_fct(group_logits, groups.argmax(-1))
        loss = self.alpha * loss + (1 - self.alpha) * groups_loss
        if not fgsm_attack:
            return loss

        # 反向传播获取 embedding 梯度
        grads = torch.autograd.grad(
            loss, embeddings, retain_graph=True, create_graph=False
        )[0]

        # FGSM 扰动：按 L2 归一化梯度
        grads_norm = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8
        adv_embeddings = embeddings + self.epsilon * grads / grads_norm

        # 对抗样本前向
        adv_features = self._forward(adv_embeddings)
        adv_logits = self.fc_labels(adv_features)
        adv_groups_logits = self.fc_groups(features)

        loss_adv = loss_fct(adv_logits, labels)
        groups_loss_adv = loss_fct(adv_groups_logits, groups.argmax(-1))
        loss_adv = self.alpha * loss_adv + (1 - self.alpha) * groups_loss_adv
        # InfoNCE 对比损失（干净特征 vs 对抗特征）
        loss_ctr = self.InfoNCE(features, adv_features, group_ids=groups)

        # 总损失：图片中的公式
        loss = (1 - lambda_fgsm) * 0.5 * (loss + loss_adv) + lambda_fgsm * loss_ctr
        return loss
