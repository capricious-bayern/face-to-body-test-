#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸编码器模块
使用MobileNetV2作为主干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as torchvision_models

class FaceEncoder(nn.Module):
    """
    人脸编码器，使用MobileNetV2作为主干网络
    输出128维L2归一化特征向量
    """
    def __init__(self, pretrained=True):
        super(FaceEncoder, self).__init__()
        # 加载MobileNetV2预训练模型
        self.backbone = torchvision_models.mobilenet_v2(pretrained=pretrained)
        # 移除分类器层
        self.backbone.classifier = nn.Identity()
        
        # 调整输出特征维度到128
        # MobileNetV2的特征提取器输出是1280维
        self.fc = nn.Linear(1280, 128)
    
    def forward(self, x):
        # 确保输入是256x256
        if x.shape[-2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        x = self.backbone.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # 使用F.adaptive_avg_pool2d进行全局平均池化
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        return x

class AttentionFaceEncoder(nn.Module):
    """
    带注意力机制的人脸编码器
    增强版本，可选使用
    """
    
    def __init__(self, pretrained=True):
        """
        初始化带注意力的人脸编码器
        
        Args:
            pretrained: 是否使用预训练权重
        """
        super(AttentionFaceEncoder, self).__init__()
        
        # 使用MobileNetV2作为主干网络
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.features = self.backbone.features
        
        # 注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [B, 3, 256, 256]
            
        Returns:
            torch.Tensor: L2归一化的特征向量 [B, 128]
        """
        # 特征提取
        features = self.features(x)  # [B, 1280, 8, 8]
        
        # 计算注意力权重
        attention_weights = self.attention(features)  # [B, 1, 8, 8]
        
        # 应用注意力
        attended_features = features * attention_weights  # [B, 1280, 8, 8]
        
        # 全局平均池化
        x = self.avgpool(attended_features)  # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]
        
        # 全连接层映射到128维
        x = self.fc(x)  # [B, 128]
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x

def create_face_encoder(encoder_type='basic', **kwargs):
    """
    创建人脸编码器
    
    Args:
        encoder_type: 编码器类型 ('basic' 或 'attention')
        **kwargs: 其他参数
        
    Returns:
        nn.Module: 人脸编码器实例
    """
    # 移除不需要的参数
    if 'image_size' in kwargs:
        del kwargs['image_size']
    
    if encoder_type == 'attention':
        return AttentionFaceEncoder(**kwargs)
    else:
        return FaceEncoder(**kwargs)

def test_face_encoder():
    """
    测试人脸编码器
    """
    print("测试人脸编码器...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建编码器
    encoder = FaceEncoder(pretrained=False).to(device)
    encoder.eval()
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"输入形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = encoder(test_input)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 验证L2归一化
    norms = torch.norm(output, p=2, dim=1)
    print(f"L2范数: {norms}")
    print(f"L2范数是否接近1: {torch.allclose(norms, torch.ones_like(norms), atol=1e-6)}")
    
    # 测试注意力版本
    print("\n测试注意力人脸编码器...")
    attention_encoder = AttentionFaceEncoder(pretrained=False).to(device)
    attention_encoder.eval()
    
    with torch.no_grad():
        attention_output = attention_encoder(test_input)
    
    print(f"注意力编码器输出形状: {attention_output.shape}")
    print(f"注意力编码器输出范围: [{attention_output.min():.4f}, {attention_output.max():.4f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("人脸编码器测试完成！")

if __name__ == "__main__":
    test_face_encoder()