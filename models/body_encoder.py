import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class BodyEncoder(nn.Module):
    """
    身体特征编码器
    基于ResNet架构，专门针对身体特征进行优化
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
        super(BodyEncoder, self).__init__()
        
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
        
        # 身体特征专用的特征投影层
        # 身体特征通常更复杂，需要更深的网络
        self.feature_projector = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim * 4),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim * 2),
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
        
        # 身体姿态感知模块
        self.pose_attention = PoseAttention(backbone_dim)
        
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
        backbone_features = self.backbone(x)  # [batch_size, backbone_dim, H, W]
        
        # 应用姿态注意力
        attended_features = self.pose_attention(backbone_features)
        
        # 全局平均池化
        pooled_features = F.adaptive_avg_pool2d(attended_features, (1, 1))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # 特征投影
        features = self.feature_projector(pooled_features)  # [batch_size, feature_dim]
        
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

class PoseAttention(nn.Module):
    """
    身体姿态注意力模块
    专门用于捕获身体的关键姿态特征
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(PoseAttention, self).__init__()
        
        self.in_channels = in_channels
        
        # 空间注意力：关注身体的关键部位
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力：关注身体特征的重要通道
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 身体部位权重学习
        self.body_part_weights = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间注意力
        spatial_att = self.spatial_conv(x)
        
        # 通道注意力
        channel_att = self.channel_attention(x)
        
        # 应用注意力权重
        attended = x * spatial_att * channel_att * self.body_part_weights
        
        return attended

class MultiScaleBodyEncoder(nn.Module):
    """
    多尺度身体编码器
    从不同尺度捕获身体特征
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 feature_dim: int = 512,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1,
                 scales: list = [1.0, 0.75, 0.5]):
        super(MultiScaleBodyEncoder, self).__init__()
        
        self.scales = scales
        self.feature_dim = feature_dim
        
        # 为每个尺度创建编码器
        self.encoders = nn.ModuleList([
            BodyEncoder(
                backbone=backbone,
                feature_dim=feature_dim // len(scales),
                pretrained=pretrained,
                dropout_rate=dropout_rate
            ) for _ in scales
        ])
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x: torch.Tensor, return_projection: bool = False):
        batch_size = x.size(0)
        multi_scale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale != 1.0:
                # 缩放输入图像
                h, w = x.size(2), x.size(3)
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
                # 恢复到原始尺寸
                scaled_x = F.interpolate(scaled_x, size=(h, w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # 提取特征
            features = self.encoders[i].get_features(scaled_x)
            multi_scale_features.append(features)
        
        # 拼接多尺度特征
        fused_features = torch.cat(multi_scale_features, dim=1)
        
        # 特征融合
        final_features = self.fusion_layer(fused_features)
        
        if return_projection:
            # 使用第一个编码器的投影头
            projections = self.encoders[0].projection_head(final_features)
            return final_features, projections
        
        return final_features

class BodyPartEncoder(nn.Module):
    """
    身体部位感知编码器
    分别处理身体的不同部位
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 feature_dim: int = 512,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1,
                 num_parts: int = 4):
        super(BodyPartEncoder, self).__init__()
        
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        
        # 基础编码器
        self.base_encoder = BodyEncoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
        
        # 身体部位分割网络
        self.part_segmentation = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_parts, 3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # 部位特征融合
        self.part_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_parts, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x: torch.Tensor, return_projection: bool = False):
        # 生成身体部位掩码
        part_masks = self.part_segmentation(x)  # [B, num_parts, H, W]
        
        part_features = []
        for i in range(self.num_parts):
            # 应用部位掩码
            mask = part_masks[:, i:i+1, :, :].expand_as(x)
            masked_input = x * mask
            
            # 提取部位特征
            part_feat = self.base_encoder.get_features(masked_input)
            part_features.append(part_feat)
        
        # 拼接所有部位特征
        concatenated_features = torch.cat(part_features, dim=1)
        
        # 特征融合
        final_features = self.part_fusion(concatenated_features)
        
        if return_projection:
            projections = self.base_encoder.projection_head(final_features)
            return final_features, projections
        
        return final_features

def create_body_encoder(model_type: str = 'basic', **kwargs) -> nn.Module:
    """
    创建身体编码器
    
    Args:
        model_type: 模型类型 ('basic', 'multiscale', 'bodypart')
        **kwargs: 其他参数
        
    Returns:
        body_encoder: 身体编码器模型
    """
    print(f"[DEBUG] create_body_encoder received kwargs: {kwargs}")
    # 移除不支持的参数
    unsupported_params = ['image_size', 'projection_dim']
    for param in unsupported_params:
        if param in kwargs:
            del kwargs[param]
    print(f"[DEBUG] create_body_encoder kwargs after removal: {kwargs}")
    if model_type == 'basic':
        return BodyEncoder(**kwargs)
    elif model_type == 'multiscale':
        return MultiScaleBodyEncoder(**kwargs)
    elif model_type == 'bodypart':
        return BodyPartEncoder(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")