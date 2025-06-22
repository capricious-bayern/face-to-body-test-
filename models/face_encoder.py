import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class FaceEncoder(nn.Module):
    """
    人脸特征编码器
    基于ResNet架构，输出固定维度的特征向量
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 feature_dim: int = 512,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1):
        """
        Args:
            backbone: 骨干网络类型 ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout比率
        """
        super(FaceEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 选择骨干网络
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加自定义的特征投影层
        self.feature_projector = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # 用于对比学习的投影头
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, 3, 256, 256]
            return_projection: 是否返回投影特征（用于对比学习）
            
        Returns:
            features: 特征向量 [batch_size, feature_dim]
            或 (features, projections) 如果return_projection=True
        """
        # 骨干网络特征提取
        backbone_features = self.backbone(x)  # [batch_size, backbone_dim, 1, 1]
        backbone_features = backbone_features.view(backbone_features.size(0), -1)  # [batch_size, backbone_dim]
        
        # 特征投影
        features = self.feature_projector(backbone_features)  # [batch_size, feature_dim]
        
        if return_projection:
            # 对比学习投影
            projections = self.projection_head(features)  # [batch_size, feature_dim//2]
            return features, projections
        
        return features
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征向量（不包含投影）"""
        return self.forward(x, return_projection=False)
    
    def get_projections(self, x: torch.Tensor) -> torch.Tensor:
        """获取投影特征（用于对比学习）"""
        _, projections = self.forward(x, return_projection=True)
        return projections

class AttentionFaceEncoder(nn.Module):
    """
    带注意力机制的人脸编码器
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 feature_dim: int = 512,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1,
                 attention_dim: int = 256):
        super(AttentionFaceEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 基础编码器
        self.base_encoder = FaceEncoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
        
        # 空间注意力模块
        self.spatial_attention = SpatialAttention()
        
        # 通道注意力模块
        if backbone in ['resnet50', 'resnet101']:
            backbone_dim = 2048
        else:
            backbone_dim = 512
            
        self.channel_attention = ChannelAttention(backbone_dim)
        
    def forward(self, x: torch.Tensor, return_projection: bool = False):
        # 获取中间特征图
        backbone = self.base_encoder.backbone
        
        # 逐层前向传播并应用注意力
        for i, layer in enumerate(backbone):
            x = layer(x)
            
            # 在最后几层应用注意力
            if i >= len(backbone) - 3:  # 在最后3层应用注意力
                if len(x.shape) == 4:  # 确保是特征图格式
                    x = self.channel_attention(x) * x
                    x = self.spatial_attention(x) * x
        
        # 全局平均池化
        x = x.view(x.size(0), -1)
        
        # 特征投影
        features = self.base_encoder.feature_projector(x)
        
        if return_projection:
            projections = self.base_encoder.projection_head(features)
            return features, projections
        
        return features

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算通道维度的最大值和平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并卷积
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        
        return self.sigmoid(attention)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # 全局平均池化和最大池化
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # 通过全连接层
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        # 相加并激活
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return attention

def create_face_encoder(model_type: str, **kwargs):
    """
    根据类型创建人脸编码器
    """
    print(f"[DEBUG] create_face_encoder received kwargs: {kwargs}")
    # 移除不支持的参数
    unsupported_params = ['image_size', 'projection_dim']
    for param in unsupported_params:
        if param in kwargs:
            del kwargs[param]
    print(f"[DEBUG] create_face_encoder kwargs after removal: {kwargs}")
    
    if model_type == "FaceEncoder":
        return FaceEncoder(**kwargs)
    elif model_type == "AttentionFaceEncoder":
        return AttentionFaceEncoder(**kwargs)
    else:
        raise ValueError(f"未知的人脸编码器类型: {model_type}")