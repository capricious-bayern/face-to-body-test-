#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸-身体特征融合和二分类器模块
用于判断人脸和身体特征是否属于同一人（伪造检测）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FaceBodyClassifier(nn.Module):
    """
    人脸-身体特征融合二分类器
    
    输入: 人脸特征 [B, 128] 和身体特征 [B, 128]
    输出: 伪造概率 [B, 1] (数值越大越可能是换脸)
    """
    
    def __init__(self, face_dim=128, body_dim=128, hidden_dim=128, dropout=0.2):
        """
        初始化分类器
        
        Args:
            face_dim: 人脸特征维度
            body_dim: 身体特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
        """
        super(FaceBodyClassifier, self).__init__()
        
        self.face_dim = face_dim
        self.body_dim = body_dim
        self.joint_dim = face_dim + body_dim  # 256
        
        # 特征融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.joint_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 输出logits，不加Sigmoid
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, face_emb, body_emb):
        """
        前向传播
        
        Args:
            face_emb: 人脸嵌入 [B, 128] (已L2归一化)
            body_emb: 身体嵌入 [B, 128] (已L2归一化)
            
        Returns:
            torch.Tensor: 伪造概率logits [B, 1]
        """
        # 特征拼接
        joint_feat = torch.cat([face_emb, body_emb], dim=1)  # [B, 256]
        
        # MLP分类
        logits = self.fusion_mlp(joint_feat)  # [B, 1]
        
        return logits
    
    def predict_proba(self, face_emb, body_emb):
        """
        预测概率（带Sigmoid）
        
        Args:
            face_emb: 人脸嵌入 [B, 128]
            body_emb: 身体嵌入 [B, 128]
            
        Returns:
            torch.Tensor: 伪造概率 [B, 1]
        """
        logits = self.forward(face_emb, body_emb)
        probs = torch.sigmoid(logits)
        return probs

class AdvancedFaceBodyClassifier(nn.Module):
    """
    增强版人脸-身体分类器
    包含注意力机制和更复杂的融合策略
    """
    
    def __init__(self, face_dim=128, body_dim=128, hidden_dim=256, num_heads=4, dropout=0.3):
        """
        初始化增强分类器
        
        Args:
            face_dim: 人脸特征维度
            body_dim: 身体特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super(AdvancedFaceBodyClassifier, self).__init__()
        
        self.face_dim = face_dim
        self.body_dim = body_dim
        
        # 特征投影层
        self.face_proj = nn.Linear(face_dim, hidden_dim)
        self.body_proj = nn.Linear(body_dim, hidden_dim)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # face + body + attention
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, face_emb, body_emb):
        """
        前向传播
        
        Args:
            face_emb: 人脸嵌入 [B, 128]
            body_emb: 身体嵌入 [B, 128]
            
        Returns:
            torch.Tensor: 伪造概率logits [B, 1]
        """
        batch_size = face_emb.shape[0]
        
        # 特征投影
        face_proj = self.face_proj(face_emb)  # [B, hidden_dim]
        body_proj = self.body_proj(body_emb)  # [B, hidden_dim]
        
        # 准备注意力输入 [B, 2, hidden_dim]
        features = torch.stack([face_proj, body_proj], dim=1)
        
        # 跨模态注意力
        attended_features, attention_weights = self.cross_attention(
            features, features, features
        )  # [B, 2, hidden_dim]
        
        # 提取注意力后的特征
        attended_face = attended_features[:, 0, :]  # [B, hidden_dim]
        attended_body = attended_features[:, 1, :]  # [B, hidden_dim]
        
        # 计算交互特征
        interaction_feat = attended_face * attended_body  # 元素级乘法
        
        # 特征拼接
        joint_feat = torch.cat([
            attended_face, 
            attended_body, 
            interaction_feat
        ], dim=1)  # [B, hidden_dim * 3]
        
        # 分类
        logits = self.fusion_layers(joint_feat)  # [B, 1]
        
        return logits

class MetricLearningClassifier(nn.Module):
    """
    基于度量学习的分类器
    通过学习人脸-身体特征的距离来判断是否匹配
    """
    
    def __init__(self, feature_dim=128, hidden_dim=64, distance_type='cosine'):
        """
        初始化度量学习分类器
        
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            distance_type: 距离类型 ('cosine', 'euclidean', 'learned')
        """
        super(MetricLearningClassifier, self).__init__()
        
        self.distance_type = distance_type
        
        if distance_type == 'learned':
            # 学习距离度量
            self.distance_net = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_distance(self, face_emb, body_emb):
        """
        计算人脸和身体特征之间的距离
        
        Args:
            face_emb: 人脸嵌入 [B, 128]
            body_emb: 身体嵌入 [B, 128]
            
        Returns:
            torch.Tensor: 距离值 [B, 1]
        """
        if self.distance_type == 'cosine':
            # 余弦相似度（已归一化，直接点积）
            similarity = torch.sum(face_emb * body_emb, dim=1, keepdim=True)
            distance = 1 - similarity  # 转换为距离
        
        elif self.distance_type == 'euclidean':
            # 欧几里得距离
            distance = torch.norm(face_emb - body_emb, p=2, dim=1, keepdim=True)
        
        elif self.distance_type == 'learned':
            # 学习的距离度量
            concat_feat = torch.cat([face_emb, body_emb], dim=1)
            distance = self.distance_net(concat_feat)
        
        else:
            raise ValueError(f"未知的距离类型: {self.distance_type}")
        
        return distance
    
    def forward(self, face_emb, body_emb):
        """
        前向传播
        
        Args:
            face_emb: 人脸嵌入 [B, 128]
            body_emb: 身体嵌入 [B, 128]
            
        Returns:
            torch.Tensor: 伪造概率logits [B, 1]
        """
        # 计算距离
        distance = self.compute_distance(face_emb, body_emb)  # [B, 1]
        
        # 分类（距离越大，越可能是伪造）
        logits = self.classifier(distance)  # [B, 1]
        
        return logits

def create_classifier(classifier_type='basic', **kwargs):
    """
    创建分类器
    
    Args:
        classifier_type: 分类器类型 ('basic', 'advanced', 'metric')
        **kwargs: 其他参数
        
    Returns:
        nn.Module: 分类器实例
    """
    if classifier_type == 'advanced':
        return AdvancedFaceBodyClassifier(**kwargs)
    elif classifier_type == 'metric':
        return MetricLearningClassifier(**kwargs)
    else:
        return FaceBodyClassifier(**kwargs)

def test_classifier():
    """
    测试分类器
    """
    print("测试FaceBodyClassifier...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 8
    feature_dim = 128
    
    # 模拟已归一化的特征
    face_emb = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1).to(device)
    body_emb = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=1).to(device)
    
    # 创建标签 (0=伪造, 1=真实)
    labels = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    print(f"Face特征形状: {face_emb.shape}")
    print(f"Body特征形状: {body_emb.shape}")
    print(f"标签: {labels}")
    
    # 测试基础分类器
    print("\n测试基础分类器...")
    classifier = FaceBodyClassifier().to(device)
    
    # 前向传播
    logits = classifier(face_emb, body_emb)
    probs = classifier.predict_proba(face_emb, body_emb)
    
    print(f"Logits形状: {logits.shape}")
    print(f"Logits范围: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"概率范围: [{probs.min():.4f}, {probs.max():.4f}]")
    
    # 测试损失计算
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits.squeeze(), labels)
    print(f"BCE损失: {loss.item():.4f}")
    
    # 测试增强分类器
    print("\n测试增强分类器...")
    advanced_classifier = AdvancedFaceBodyClassifier().to(device)
    
    advanced_logits = advanced_classifier(face_emb, body_emb)
    print(f"增强分类器Logits形状: {advanced_logits.shape}")
    print(f"增强分类器Logits范围: [{advanced_logits.min():.4f}, {advanced_logits.max():.4f}]")
    
    # 测试度量学习分类器
    print("\n测试度量学习分类器...")
    metric_classifier = MetricLearningClassifier(distance_type='cosine').to(device)
    
    metric_logits = metric_classifier(face_emb, body_emb)
    print(f"度量分类器Logits形状: {metric_logits.shape}")
    print(f"度量分类器Logits范围: [{metric_logits.min():.4f}, {metric_logits.max():.4f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"\n基础分类器参数量: {total_params:,}")
    
    advanced_params = sum(p.numel() for p in advanced_classifier.parameters())
    print(f"增强分类器参数量: {advanced_params:,}")
    
    metric_params = sum(p.numel() for p in metric_classifier.parameters())
    print(f"度量分类器参数量: {metric_params:,}")
    
    # 测试梯度
    print("\n测试梯度计算...")
    face_emb.requires_grad_(True)
    body_emb.requires_grad_(True)
    
    logits = classifier(face_emb, body_emb)
    loss = criterion(logits.squeeze(), labels)
    loss.backward()
    
    print(f"Face特征梯度范数: {face_emb.grad.norm().item():.6f}")
    print(f"Body特征梯度范数: {body_emb.grad.norm().item():.6f}")
    
    print("分类器测试完成！")

if __name__ == "__main__":
    test_classifier()