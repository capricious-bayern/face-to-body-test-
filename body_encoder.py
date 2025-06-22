#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
身体编码器模块
使用EfficientNet-B0作为主干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

class BodyEncoder(nn.Module):
    """
    身体编码器，可选择MobileNetV2或EfficientNet-B0作为主干网络
    输出128维L2归一化特征向量
    """
    def __init__(self, backbone='mobilenet_v2', pretrained=True):
        super(BodyEncoder, self).__init__()
        self.backbone_name = backbone
        
        if backbone == 'mobilenet_v2':
            print("使用MobileNetV2作为身体编码器主干网络")
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            self.output_dim = 1280
        elif backbone == 'efficientnet-b0':
            print("使用EfficientNet-B0作为身体编码器主干网络")
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            self.backbone._fc = nn.Identity()
            self.output_dim = 1280
        else:
            raise ValueError(f"不支持的主干网络: {backbone}")
        
        # 调整输出特征维度到128
        self.fc = nn.Linear(self.output_dim, 128)
    
    def forward(self, x):
        # 确保输入是256x256
        if x.shape[-2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        if self.backbone_name == 'mobilenet_v2':
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
        elif self.backbone_name == 'efficientnet-b0':
            x = self.backbone.extract_features(x)
            x = self.backbone._avg_pooling(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        return x

class PoseAttention(nn.Module):
    """
    姿态注意力模块
    用于关注身体的关键部位
    """
    
    def __init__(self, in_channels):
        super(PoseAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class MultiScaleBodyEncoder(nn.Module):
    """
    多尺度身体编码器，结合不同尺度的特征
    """
    def __init__(self, backbone='mobilenet_v2', pretrained=True):
        super(MultiScaleBodyEncoder, self).__init__()
        self.backbone_name = backbone

        if backbone == 'mobilenet_v2':
            print("使用MobileNetV2作为多尺度身体编码器主干网络")
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            # MobileNetV2的特征层
            self.features_small = self.backbone.features[:7]  # block 0-6, output 96 channels
            self.features_medium = self.backbone.features[7:14] # block 7-13, output 320 channels
            self.features_large = self.backbone.features[14:] # block 14-18, output 1280 channels
            self.fc_small = nn.Linear(96, 128)
            self.fc_medium = nn.Linear(320, 128)
            self.fc_large = nn.Linear(1280, 128)
        elif backbone == 'efficientnet-b0':
            print("使用EfficientNet-B0作为多尺度身体编码器主干网络")
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            # EfficientNet-B0的特征层
            self.features_small = self.backbone.extract_features[:3] # block 0-2, output 24 channels
            self.features_medium = self.backbone.extract_features[3:5] # block 3-4, output 40 channels
            self.features_large = self.backbone.extract_features[5:] # block 5-8, output 1280 channels
            self.fc_small = nn.Linear(24, 128)
            self.fc_medium = nn.Linear(40, 128)
            self.fc_large = nn.Linear(1280, 128)
        else:
            raise ValueError(f"不支持的主干网络: {backbone}")

    def forward(self, x):
        # 确保输入是256x256
        if x.shape[-2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # 提取多尺度特征
        x_small = self.features_small(x)
        x_medium = self.features_medium(x_small)
        x_large = self.features_large(x_medium)

        # 全局平均池化
        x_small = F.adaptive_avg_pool2d(x_small, (1, 1)).flatten(1)
        x_medium = F.adaptive_avg_pool2d(x_medium, (1, 1)).flatten(1)
        x_large = F.adaptive_avg_pool2d(x_large, (1, 1)).flatten(1)

        # 映射到统一维度
        feat_small = self.fc_small(x_small)
        feat_medium = self.fc_medium(x_medium)
        feat_large = self.fc_large(x_large)

        # 融合特征 (简单相加)
        fused_features = feat_small + feat_medium + feat_large

        # L2归一化
        fused_features = F.normalize(fused_features, p=2, dim=1)
        return fused_features

class BodyPartEncoder(nn.Module):
    """
    身体部位编码器
    专门处理身体不同部位的特征
    """
    
    def __init__(self, backbone='mobilenet_v2', pretrained=True):
        super(BodyPartEncoder, self).__init__()
        
        # 使用轻量级网络
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # 身体部位特定的卷积层
        self.part_conv = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 主干特征提取
        x = self.backbone.features(x)
        
        # 身体部位特定处理
        x = self.part_conv(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类器
        x = self.classifier(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x

def create_body_encoder(encoder_type='basic', **kwargs):
    """
    创建身体编码器
    
    Args:
        encoder_type: 编码器类型 ('basic', 'multiscale', 'bodypart')
        **kwargs: 其他参数
        
    Returns:
        nn.Module: 身体编码器实例
    """
    # 移除不需要的参数
    if 'image_size' in kwargs:
        del kwargs['image_size']
    
    if encoder_type == 'multiscale':
        return MultiScaleBodyEncoder(**kwargs)
    elif encoder_type == 'bodypart':
        return BodyPartEncoder(**kwargs)
    else:
        return BodyEncoder(**kwargs)

def test_body_encoder():
    """
    测试身体编码器
    """
    print("测试身体编码器...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建编码器
    encoder = BodyEncoder(backbone='mobilenet_v2', pretrained=False).to(device)
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
    
    # 测试多尺度版本
    print("\n测试多尺度身体编码器...")
    multiscale_encoder = MultiScaleBodyEncoder(backbone='mobilenet_v2', pretrained=False).to(device)
    multiscale_encoder.eval()
    
    with torch.no_grad():
        multiscale_output = multiscale_encoder(test_input)
    
    print(f"多尺度编码器输出形状: {multiscale_output.shape}")
    print(f"多尺度编码器输出范围: [{multiscale_output.min():.4f}, {multiscale_output.max():.4f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("身体编码器测试完成！")

if __name__ == "__main__":
    test_body_encoder()