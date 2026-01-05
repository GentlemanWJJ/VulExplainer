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
        self.num_blocks=3
        self.dim_channel=256
        self.epsilon = 1e-3  # FGSM扰动系数
        self.tau = getattr(args, "contrastive_tau", 0.5)  # 对比学习温度系数τ

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

        self.category_head = nn.Linear(self.dim_channel, num_class)
        self.class_head = nn.Linear(self.dim_channel, num_class)
        self.variant_head = nn.Linear(self.dim_channel, num_class)
        self.base_head = nn.Linear(self.dim_channel, num_class)
        self.deprecated_head = nn.Linear(self.dim_channel, num_class)

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

    def _contrastive_loss(self, clean_feat, adv_feat):
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

    def forward(self, input_ids,groups, labels, fgsm_attack=False, lambda_fgsm=0.1):
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

        # 正常前向（干净样本）
        features = self._forward(embeddings)
        logits = self.fc(features)

        category_logits = self.category_head(features)
        class_logits = self.class_head(features)
        variant_logits = self.variant_head(features)
        base_logits = self.base_head(features)
        deprecated_logits = self.deprecated_head(features)
        logits = (
            torch.empty(category_logits.shape[0], category_logits.shape[1])
            .float()
            .to(self.args.device)
        )

        for i in range(len(groups)):
            idx=torch.argmax(groups[i], dim=-1)
            if idx.item() == 0:
                logits[i, :] = category_logits[i]
            elif idx.item() == 1:
                logits[i, :] = class_logits[i]
            elif idx.item() == 2:
                logits[i, :] = variant_logits[i]
            elif idx.item() == 3:
                logits[i, :] = base_logits[i]
            elif idx.item() == 4:
                logits[i, :] = deprecated_logits[i]
            elif idx.item() == 5:
                logits[i, :] = deprecated_logits[i]
        if labels is None or not fgsm_attack:
            if labels is None:
                return F.softmax(logits, dim=-1)
            # 仅使用干净样本的交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            # groups_loss = loss_fct(groups_logits, groups_one_hot)
            return loss

        # # 训练阶段 + 启用 FGSM 对抗与对比学习
        # loss_fct = nn.CrossEntropyLoss()
        # loss_clean = loss_fct(logits, labels)

        # # 反向传播获取 embedding 梯度
        # grads = torch.autograd.grad(
        #     loss_clean, embeddings, retain_graph=True, create_graph=False
        # )[0]

        # # FGSM 扰动：按 L2 归一化梯度
        # grads_norm = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8
        # adv_embeddings = embeddings + self.epsilon * grads / grads_norm

        # # 对抗样本前向
        # adv_features = self._feature_forward(adv_embeddings)
        # adv_logits = self.fc(adv_features)
        # loss_adv = loss_fct(adv_logits, labels)

        # # InfoNCE 对比损失（干净特征 vs 对抗特征）
        # loss_ctr = self._contrastive_loss(features, adv_features)

        # # 总损失：图片中的公式
        # total_loss = (1 - lambda_fgsm) * 0.5 * (loss_clean + loss_adv) + lambda_fgsm * loss_ctr
        # return total_loss


class TextCNN(nn.Module):

    def __init__(
        self,
        encoder,
        tokenizer,
        num_class,
        args,
    ):
        super(TextCNN, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        emb_dim = encoder.config.hidden_size
        dim_channel=100
        kernel_wins = [3, 4, 5]
        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins]
        )
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        # FC layer
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)

    def forward(self, input_ids, labels=None, return_hidden_state=False):
        emb_x = self.encoder.embeddings(input_ids)
        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id)
            .unsqueeze(-1)
            .expand(input_ids.shape[0], input_ids.shape[1], 768)
        )
        emb_x = emb_x * attention_mask
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logit, labels)
            return loss
        prob = torch.softmax(logit, dim=-1)
        if return_hidden_state:
            return fc_x
        else:
            return prob
