import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FaceBodyClassifier(nn.Module):
    """
    人脸-身体匹配分类器
    融合人脸和身体特征进行真实/伪造二分类
    """
    
    def __init__(self, 
                 face_feature_dim: int = 512,
                 body_feature_dim: int = 512,
                 hidden_dims: list = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 fusion_method: str = 'concat'):
        """
        Args:
            face_feature_dim: 人脸特征维度
            body_feature_dim: 身体特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            fusion_method: 特征融合方法 ('concat', 'add', 'multiply', 'attention')
        """
        super(FaceBodyClassifier, self).__init__()
        
        self.face_feature_dim = face_feature_dim
        self.body_feature_dim = body_feature_dim
        self.fusion_method = fusion_method
        
        # 特征融合
        if fusion_method == 'concat':
            self.fusion_dim = face_feature_dim + body_feature_dim
        elif fusion_method in ['add', 'multiply']:
            assert face_feature_dim == body_feature_dim, "Features must have same dim for add/multiply"
            self.fusion_dim = face_feature_dim
        elif fusion_method == 'attention':
            self.fusion_dim = max(face_feature_dim, body_feature_dim)
            self.attention_fusion = AttentionFusion(face_feature_dim, body_feature_dim, self.fusion_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 构建分类网络
        layers = []
        input_dim = self.fusion_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 2))  # 二分类
        
        self.classifier = nn.Sequential(*layers)
        
        # 特征重要性权重（可学习）
        self.feature_weights = nn.Parameter(torch.ones(2))  # [face_weight, body_weight]
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            face_features: 人脸特征 [batch_size, face_feature_dim]
            body_features: 身体特征 [batch_size, body_feature_dim]
            
        Returns:
            logits: 分类logits [batch_size, 2]
        """
        # 应用特征权重
        weighted_face = face_features * self.feature_weights[0]
        weighted_body = body_features * self.feature_weights[1]
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused_features = torch.cat([weighted_face, weighted_body], dim=1)
        elif self.fusion_method == 'add':
            fused_features = weighted_face + weighted_body
        elif self.fusion_method == 'multiply':
            fused_features = weighted_face * weighted_body
        elif self.fusion_method == 'attention':
            fused_features = self.attention_fusion(weighted_face, weighted_body)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict_proba(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        logits = self.forward(face_features, body_features)
        return F.softmax(logits, dim=1)
    
    def predict(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        logits = self.forward(face_features, body_features)
        return torch.argmax(logits, dim=1)

class AttentionFusion(nn.Module):
    """
    注意力特征融合模块
    """
    
    def __init__(self, face_dim: int, body_dim: int, output_dim: int):
        super(AttentionFusion, self).__init__()
        
        # 特征投影
        self.face_proj = nn.Linear(face_dim, output_dim)
        self.body_proj = nn.Linear(body_dim, output_dim)
        
        # 注意力计算
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        # 特征投影
        face_proj = self.face_proj(face_features)
        body_proj = self.body_proj(body_features)
        
        # 计算注意力权重
        concat_features = torch.cat([face_proj, body_proj], dim=1)
        attention_weights = self.attention(concat_features)  # [batch_size, 2]
        
        # 加权融合
        fused = (attention_weights[:, 0:1] * face_proj + 
                attention_weights[:, 1:2] * body_proj)
        
        return fused

class MultiModalClassifier(nn.Module):
    """
    多模态分类器
    支持多种特征融合策略
    """
    
    def __init__(self, 
                 face_feature_dim: int = 512,
                 body_feature_dim: int = 512,
                 hidden_dims: list = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 use_cross_attention: bool = True):
        super(MultiModalClassifier, self).__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # 交叉注意力模块
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(face_feature_dim, body_feature_dim)
            fusion_dim = face_feature_dim + body_feature_dim
        else:
            fusion_dim = face_feature_dim + body_feature_dim
        
        # 特征增强
        self.face_enhancer = FeatureEnhancer(face_feature_dim)
        self.body_enhancer = FeatureEnhancer(body_feature_dim)
        
        # 分类网络
        layers = []
        input_dim = fusion_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 2))
        self.classifier = nn.Sequential(*layers)
        
        # 辅助损失分支
        self.aux_face_classifier = nn.Linear(face_feature_dim, 2)
        self.aux_body_classifier = nn.Linear(body_feature_dim, 2)
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor,
                return_aux: bool = False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # 特征增强
        enhanced_face = self.face_enhancer(face_features)
        enhanced_body = self.body_enhancer(body_features)
        
        # 交叉注意力
        if self.use_cross_attention:
            attended_face, attended_body = self.cross_attention(enhanced_face, enhanced_body)
        else:
            attended_face, attended_body = enhanced_face, enhanced_body
        
        # 特征融合
        fused_features = torch.cat([attended_face, attended_body], dim=1)
        
        # 主分类
        main_logits = self.classifier(fused_features)
        
        if return_aux:
            # 辅助分类
            aux_face_logits = self.aux_face_classifier(attended_face)
            aux_body_logits = self.aux_body_classifier(attended_body)
            return main_logits, (aux_face_logits, aux_body_logits)
        
        return main_logits

class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    """
    
    def __init__(self, face_dim: int, body_dim: int, hidden_dim: int = 256):
        super(CrossModalAttention, self).__init__()
        
        self.face_dim = face_dim
        self.body_dim = body_dim
        self.hidden_dim = hidden_dim
        
        # 查询、键、值投影
        self.face_query = nn.Linear(face_dim, hidden_dim)
        self.face_key = nn.Linear(face_dim, hidden_dim)
        self.face_value = nn.Linear(face_dim, hidden_dim)
        
        self.body_query = nn.Linear(body_dim, hidden_dim)
        self.body_key = nn.Linear(body_dim, hidden_dim)
        self.body_value = nn.Linear(body_dim, hidden_dim)
        
        # 输出投影
        self.face_out = nn.Linear(hidden_dim, face_dim)
        self.body_out = nn.Linear(hidden_dim, body_dim)
        
        self.scale = hidden_dim ** -0.5
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Face attends to Body
        face_q = self.face_query(face_features)
        body_k = self.body_key(body_features)
        body_v = self.body_value(body_features)
        
        face_attention = torch.softmax(torch.matmul(face_q, body_k.T) * self.scale, dim=-1)
        attended_face = self.face_out(torch.matmul(face_attention, body_v))
        
        # Body attends to Face
        body_q = self.body_query(body_features)
        face_k = self.face_key(face_features)
        face_v = self.face_value(face_features)
        
        body_attention = torch.softmax(torch.matmul(body_q, face_k.T) * self.scale, dim=-1)
        attended_body = self.body_out(torch.matmul(body_attention, face_v))
        
        # 残差连接
        attended_face = attended_face + face_features
        attended_body = attended_body + body_features
        
        return attended_face, attended_body

class FeatureEnhancer(nn.Module):
    """
    特征增强模块
    """
    
    def __init__(self, feature_dim: int, enhancement_ratio: float = 0.25):
        super(FeatureEnhancer, self).__init__()
        
        hidden_dim = int(feature_dim * enhancement_ratio)
        
        self.enhancer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhancement = self.enhancer(x)
        return x * enhancement

class EnsembleClassifier(nn.Module):
    """
    集成分类器
    结合多个分类器的预测结果
    """
    
    def __init__(self, classifiers: list, weights: Optional[list] = None):
        super(EnsembleClassifier, self).__init__()
        
        self.classifiers = nn.ModuleList(classifiers)
        
        if weights is None:
            weights = [1.0 / len(classifiers)] * len(classifiers)
        
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        
    def forward(self, face_features: torch.Tensor, body_features: torch.Tensor) -> torch.Tensor:
        predictions = []
        
        for classifier in self.classifiers:
            pred = classifier(face_features, body_features)
            predictions.append(F.softmax(pred, dim=1))
        
        # 加权平均
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weighted_predictions.append(self.weights[i] * pred)
        
        ensemble_pred = torch.stack(weighted_predictions).sum(dim=0)
        
        # 转换回logits
        ensemble_logits = torch.log(ensemble_pred + 1e-8)
        
        return ensemble_logits

def create_classifier(classifier_type: str = 'basic', **kwargs) -> nn.Module:
    """
    创建分类器
    
    Args:
        classifier_type: 分类器类型 ('basic', 'multimodal', 'ensemble')
        **kwargs: 其他参数
        
    Returns:
        classifier: 分类器模型
    """
    if classifier_type == 'basic':
        return FaceBodyClassifier(**kwargs)
    elif classifier_type == 'multimodal':
        return MultiModalClassifier(**kwargs)
    elif classifier_type == 'ensemble':
        return EnsembleClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")