import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class SupConLoss(nn.Module):
    """
    有监督对比损失 (Supervised Contrastive Loss)
    基于SupCon论文实现，适用于身份感知的face-body匹配
    """
    
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        """
        Args:
            temperature: 温度参数，控制分布的尖锐程度
            contrast_mode: 对比模式 ('one', 'all')
            base_temperature: 基础温度参数
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算有监督对比损失
        
        Args:
            features: 特征向量 [bsz, n_views, feature_dim] 或 [bsz, feature_dim]
            labels: 标签 [bsz]
            mask: 对比掩码 [bsz, bsz]
            
        Returns:
            loss: 对比损失值
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 构建掩码
        mask = mask.repeat(anchor_count, contrast_count)
        # 移除自身对比
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算平均log似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数
    用于face-body特征对比学习
    """
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            face_features: 人脸特征 [batch_size, feature_dim]
            body_features: 身体特征 [batch_size, feature_dim]
            
        Returns:
            loss: InfoNCE损失值
        """
        batch_size = face_features.shape[0]
        
        # L2归一化
        face_features = F.normalize(face_features, dim=1)
        body_features = F.normalize(body_features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(face_features, body_features.T) / self.temperature
        
        # 正样本标签（对角线）
        labels = torch.arange(batch_size).to(face_features.device)
        
        # 计算损失（双向）
        loss_face_to_body = self.criterion(similarity_matrix, labels)
        loss_body_to_face = self.criterion(similarity_matrix.T, labels)
        
        return (loss_face_to_body + loss_body_to_face) / 2

class IdentityAwareContrastiveLoss(nn.Module):
    """
    身份感知对比损失
    结合身份信息的对比学习损失函数
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5,
                 identity_weight: float = 1.0, contrastive_weight: float = 1.0):
        super(IdentityAwareContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.identity_weight = identity_weight
        self.contrastive_weight = contrastive_weight
        
        self.infonce_loss = InfoNCELoss(temperature)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor,
                identity_labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算身份感知对比损失
        
        Args:
            face_features: 人脸特征 [batch_size, feature_dim]
            body_features: 身体特征 [batch_size, feature_dim]
            identity_labels: 身份标签 [batch_size]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # InfoNCE损失
        infonce_loss = self.infonce_loss(face_features, body_features)
        
        # 身份三元组损失
        identity_loss = self._compute_identity_triplet_loss(
            face_features, body_features, identity_labels
        )
        
        # 总损失
        total_loss = (self.contrastive_weight * infonce_loss + 
                     self.identity_weight * identity_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'infonce_loss': infonce_loss.item(),
            'identity_loss': identity_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_identity_triplet_loss(self, face_features: torch.Tensor,
                                     body_features: torch.Tensor,
                                     identity_labels: torch.Tensor) -> torch.Tensor:
        """
        计算基于身份的三元组损失
        """
        batch_size = face_features.shape[0]
        device = face_features.device
        
        # 构建三元组
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            anchor_id = identity_labels[i]
            
            # 找到同一身份的正样本
            positive_mask = (identity_labels == anchor_id)
            positive_indices = torch.where(positive_mask)[0]
            
            # 找到不同身份的负样本
            negative_mask = (identity_labels != anchor_id)
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) > 1 and len(negative_indices) > 0:
                # 选择正样本（除了自己）
                pos_idx = positive_indices[positive_indices != i][0]
                # 选择负样本
                neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))][0]
                
                anchors.append(face_features[i])
                positives.append(body_features[pos_idx])
                negatives.append(body_features[neg_idx])
        
        if len(anchors) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        
        return self.triplet_loss(anchors, positives, negatives)

class AdaptiveContrastiveLoss(nn.Module):
    """
    自适应对比损失
    根据训练进度动态调整损失权重
    """
    
    def __init__(self, temperature: float = 0.07, 
                 warmup_epochs: int = 10, total_epochs: int = 100):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        self.infonce_loss = InfoNCELoss(temperature)
        self.supcon_loss = SupConLoss(temperature)
        
    def set_epoch(self, epoch: int):
        """设置当前训练轮数"""
        self.current_epoch = epoch
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算自适应对比损失
        """
        # 计算权重
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段，主要使用InfoNCE
            infonce_weight = 1.0
            supcon_weight = 0.1
        else:
            # 正常训练阶段，逐渐增加SupCon权重
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            infonce_weight = 1.0 - 0.5 * progress
            supcon_weight = 0.1 + 0.9 * progress
        
        # InfoNCE损失
        infonce_loss = self.infonce_loss(face_features, body_features)
        
        # SupCon损失
        if labels is not None:
            # 将face和body特征组合
            combined_features = torch.stack([face_features, body_features], dim=1)
            supcon_loss = self.supcon_loss(combined_features, labels)
        else:
            supcon_loss = torch.tensor(0.0, device=face_features.device)
        
        total_loss = infonce_weight * infonce_loss + supcon_weight * supcon_loss
        
        return total_loss

class FocalContrastiveLoss(nn.Module):
    """
    焦点对比损失
    关注困难样本的对比学习
    """
    
    def __init__(self, temperature: float = 0.07, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        """
        计算焦点对比损失
        """
        batch_size = face_features.shape[0]
        device = face_features.device
        
        # L2归一化
        face_features = F.normalize(face_features, dim=1)
        body_features = F.normalize(body_features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(face_features, body_features.T) / self.temperature
        
        # 计算概率
        probs = F.softmax(similarity_matrix, dim=1)
        
        # 正样本概率（对角线）
        positive_probs = torch.diag(probs)
        
        # 焦点权重
        focal_weights = self.alpha * (1 - positive_probs) ** self.gamma
        
        # 负对数似然
        nll_loss = -torch.log(positive_probs + 1e-8)
        
        # 焦点损失
        focal_loss = focal_weights * nll_loss
        
        return focal_loss.mean()

def create_contrastive_loss(loss_type: str = 'infonce', **kwargs) -> nn.Module:
    """
    创建对比损失函数
    
    Args:
        loss_type: 损失类型 ('infonce', 'supcon', 'identity_aware', 'adaptive', 'focal')
        **kwargs: 其他参数
        
    Returns:
        loss_fn: 损失函数
    """
    if loss_type == 'infonce':
        return InfoNCELoss(**kwargs)
    elif loss_type == 'supcon':
        return SupConLoss(**kwargs)
    elif loss_type == 'identity_aware':
        return IdentityAwareContrastiveLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveContrastiveLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")