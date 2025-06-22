#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有监督对比学习损失函数
Supervised Contrastive Loss for Face-Body Consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupConLoss(nn.Module):
    """
    有监督对比学习损失函数
    
    参考论文: Supervised Contrastive Learning (Khosla et al.)
    用于训练人脸和身体特征的一致性
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        初始化SupConLoss
        
        Args:
            temperature: 温度系数，控制分布的尖锐程度
            contrast_mode: 对比模式 ('all' 或 'one')
            base_temperature: 基础温度系数
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        计算有监督对比学习损失
        
        Args:
            features: 特征张量 [bsz, n_views, feature_dim] 或 [bsz, feature_dim]
                     对于face-body任务，通常是 [2*bsz, feature_dim] (face和body特征拼接)
            labels: 标签张量 [bsz] 表示每个样本的身份ID
            mask: 对比掩码 [bsz, bsz] (可选)
            
        Returns:
            torch.Tensor: 标量损失值
        """
        device = features.device
        
        # 处理输入特征的维度
        if len(features.shape) < 3:
            features = features.unsqueeze(1)  # [bsz, 1, feature_dim]
        
        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('不能同时指定labels和mask')
        elif labels is None and mask is None:
            # 如果没有标签，创建单位矩阵作为掩码（每个样本只与自己对比）
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('标签数量与批次大小不匹配')
            
            # 创建标签掩码：相同标签的样本为正样本对
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]  # 视图数量
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, feature_dim]
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # 只使用第一个视图作为anchor
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # 使用所有视图作为anchor
            anchor_count = contrast_count
        else:
            raise ValueError('未知的对比模式: {}'.format(self.contrast_mode))
        
        # 计算相似度矩阵
        # anchor_feature: [anchor_count*bsz, feature_dim]
        # contrast_feature: [contrast_count*bsz, feature_dim]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )  # [anchor_count*bsz, contrast_count*bsz]
        
        # 数值稳定性：减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 构建掩码
        mask = mask.repeat(anchor_count, contrast_count)
        
        # 创建对角掩码，排除自己与自己的对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算正样本对的对数概率（数值稳定版本）
        exp_logits = torch.exp(logits) * logits_mask
        
        # 添加小的epsilon防止log(0)
        eps = 1e-8
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)
        
        # 计算每个anchor的正样本对的平均对数概率
        mask_sum = mask.sum(1)
        # 防止除零
        mask_sum = torch.clamp(mask_sum, min=eps)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # 过滤掉无效的损失值（没有正样本对的情况）
        valid_mask = mask.sum(1) > 0
        if valid_mask.sum() == 0:
            # 如果没有有效的正样本对，返回零损失
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        mean_log_prob_pos = mean_log_prob_pos[valid_mask]
        
        # 损失计算
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        # 检查NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        return loss

class FaceBodySupConLoss(nn.Module):
    """
    专门用于Face-Body对比学习的损失函数
    """
    
    def __init__(self, temperature=0.07, alpha=1.0, beta=1.0):
        """
        初始化Face-Body对比损失
        
        Args:
            temperature: 温度系数
            alpha: face内部对比的权重
            beta: face-body跨模态对比的权重
        """
        super(FaceBodySupConLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.supcon_loss = SupConLoss(temperature=temperature)
    
    def forward(self, face_embeddings, body_embeddings, identity_labels):
        """
        计算Face-Body对比学习损失
        
        Args:
            face_embeddings: 人脸特征 [B, 128]
            body_embeddings: 身体特征 [B, 128]
            identity_labels: 身份标签 [B]
            
        Returns:
            dict: 包含各种损失的字典
        """
        batch_size = face_embeddings.shape[0]
        device = face_embeddings.device
        
        # 检查输入有效性
        if batch_size == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'total_loss': zero_loss,
                'face_loss': zero_loss,
                'body_loss': zero_loss,
                'cross_modal_loss': zero_loss
            }
        
        # 1. Face内部对比损失
        try:
            face_loss = self.supcon_loss(face_embeddings, identity_labels)
        except Exception:
            face_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 2. Body内部对比损失
        try:
            body_loss = self.supcon_loss(body_embeddings, identity_labels)
        except Exception:
            body_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 3. Face-Body跨模态对比损失
        try:
            # 将face和body特征拼接，每个身份有两个视图
            combined_features = torch.stack([face_embeddings, body_embeddings], dim=1)  # [B, 2, 128]
            cross_modal_loss = self.supcon_loss(combined_features, identity_labels)
        except Exception:
            cross_modal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 检查NaN
        if torch.isnan(face_loss):
            face_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(body_loss):
            body_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(cross_modal_loss):
            cross_modal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 总损失
        total_loss = (
            self.alpha * (face_loss + body_loss) / 2 +
            self.beta * cross_modal_loss
        )
        
        # 最终检查
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {
            'total_loss': total_loss,
            'face_loss': face_loss,
            'body_loss': body_loss,
            'cross_modal_loss': cross_modal_loss
        }

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数
    另一种对比学习损失的实现
    """
    
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, query, positive_key, negative_keys):
        """
        计算InfoNCE损失
        
        Args:
            query: 查询特征 [B, D]
            positive_key: 正样本特征 [B, D]
            negative_keys: 负样本特征 [B, N, D]
            
        Returns:
            torch.Tensor: 损失值
        """
        # 计算正样本相似度
        pos_sim = torch.sum(query * positive_key, dim=1) / self.temperature  # [B]
        
        # 计算负样本相似度
        neg_sim = torch.bmm(negative_keys, query.unsqueeze(2)).squeeze(2) / self.temperature  # [B, N]
        
        # 拼接正负样本相似度
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        
        # 正样本的标签是0
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss

def test_supcon_loss():
    """
    测试对比学习损失函数
    """
    print("测试SupConLoss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 8
    feature_dim = 128
    num_identities = 4
    
    # 模拟face和body特征
    face_embeddings = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1).to(device)
    body_embeddings = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1).to(device)
    
    # 创建身份标签（每个身份有2个样本）
    identity_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(device)
    
    print(f"Face特征形状: {face_embeddings.shape}")
    print(f"Body特征形状: {body_embeddings.shape}")
    print(f"身份标签: {identity_labels}")
    
    # 测试基础SupConLoss
    supcon_loss = SupConLoss(temperature=0.07).to(device)
    
    # 测试face特征
    face_loss = supcon_loss(face_embeddings, identity_labels)
    print(f"Face SupCon损失: {face_loss.item():.4f}")
    
    # 测试body特征
    body_loss = supcon_loss(body_embeddings, identity_labels)
    print(f"Body SupCon损失: {body_loss.item():.4f}")
    
    # 测试跨模态对比
    combined_features = torch.stack([face_embeddings, body_embeddings], dim=1)  # [B, 2, 128]
    cross_modal_loss = supcon_loss(combined_features, identity_labels)
    print(f"跨模态SupCon损失: {cross_modal_loss.item():.4f}")
    
    # 测试Face-Body专用损失
    print("\n测试FaceBodySupConLoss...")
    fb_loss = FaceBodySupConLoss(temperature=0.07).to(device)
    
    loss_dict = fb_loss(face_embeddings, body_embeddings, identity_labels)
    
    print(f"总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"Face损失: {loss_dict['face_loss'].item():.4f}")
    print(f"Body损失: {loss_dict['body_loss'].item():.4f}")
    print(f"跨模态损失: {loss_dict['cross_modal_loss'].item():.4f}")
    
    # 测试梯度
    print("\n测试梯度计算...")
    face_embeddings.requires_grad_(True)
    body_embeddings.requires_grad_(True)
    
    loss_dict = fb_loss(face_embeddings, body_embeddings, identity_labels)
    total_loss = loss_dict['total_loss']
    total_loss.backward()
    
    print(f"Face特征梯度范数: {face_embeddings.grad.norm().item():.6f}")
    print(f"Body特征梯度范数: {body_embeddings.grad.norm().item():.6f}")
    
    print("SupConLoss测试完成！")

if __name__ == "__main__":
    test_supcon_loss()