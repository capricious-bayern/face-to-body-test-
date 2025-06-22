#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练主循环模块
用于联合训练人脸-身体伪造检测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceBodyTrainer:
    """
    人脸-身体伪造检测模型训练器
    """
    
    def __init__(self, 
                 face_encoder, 
                 body_encoder, 
                 classifier, 
                 supcon_loss, 
                 device='cuda',
                 contrastive_weight=1.0,
                 classification_weight=0.5):
        """
        初始化训练器
        
        Args:
            face_encoder: 人脸编码器
            body_encoder: 身体编码器
            classifier: 分类器
            supcon_loss: 对比学习损失函数
            device: 计算设备
            contrastive_weight: 对比学习损失权重
            classification_weight: 分类损失权重
        """
        self.device = torch.device(device)
        
        # 模型
        self.face_encoder = face_encoder.to(self.device)
        self.body_encoder = body_encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        
        # 损失函数
        self.supcon_loss = supcon_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 损失权重
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        
        # 优化器（稍后设置）
        self.optimizer = None
        self.scheduler = None
        
        # 训练统计
        self.train_stats = defaultdict(list)
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
    
    def setup_optimizer(self, lr=1e-3, weight_decay=1e-4, scheduler_type='cosine', **kwargs):
        """
        设置优化器和学习率调度器
        
        Args:
            lr: 学习率
            weight_decay: 权重衰减
            scheduler_type: 调度器类型
            **kwargs: 其他参数
        """
        # 收集所有模型参数
        all_params = list(self.face_encoder.parameters()) + \
                    list(self.body_encoder.parameters()) + \
                    list(self.classifier.parameters())
        
        # 创建优化器
        self.optimizer = optim.Adam(
            all_params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        
        # 创建学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )
        else:
            self.scheduler = None
        
        logger.info(f"优化器设置完成: lr={lr}, weight_decay={weight_decay}, scheduler={scheduler_type}")
    
    def train_one_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            dict: 训练统计信息
        """
        # 设置为训练模式
        self.face_encoder.train()
        self.body_encoder.train()
        self.classifier.train()
        
        # 统计变量
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_classification_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            face_images = batch['face'].to(self.device)  # [B, 3, 256, 256]
            body_images = batch['body'].to(self.device)  # [B, 3, 256, 256]
            labels = batch['label'].to(self.device).float()  # [B] 0=伪造, 1=真实
            identity_ids = batch['identity'].to(self.device)  # [B] 身份ID
            
            batch_size = face_images.shape[0]
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            face_emb = self.face_encoder(face_images)  # [B, 128]
            body_emb = self.body_encoder(body_images)  # [B, 128]
            
            # 1. 计算对比学习损失
            contrastive_loss = self._compute_contrastive_loss(face_emb, body_emb, identity_ids)
            
            # 2. 计算分类损失
            classification_logits = self.classifier(face_emb, body_emb)  # [B, 1]
            classification_loss = self.bce_loss(classification_logits.squeeze(), labels)
            
            # 3. 总损失
            total_batch_loss = (
                self.contrastive_weight * contrastive_loss +
                self.classification_weight * classification_loss
            )
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(
                list(self.face_encoder.parameters()) + 
                list(self.body_encoder.parameters()) + 
                list(self.classifier.parameters()),
                max_norm=1.0
            )
            
            # 优化器步进
            self.optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                predictions = torch.sigmoid(classification_logits.squeeze()) > 0.5
                correct_predictions += (predictions == labels.bool()).sum().item()
                total_samples += batch_size
            
            # 累计损失
            total_loss += total_batch_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_classification_loss += classification_loss.item()
            
            # 更新进度条
            postfix_dict = {
                'Loss': f'{total_batch_loss.item():.4f}',
                'ConLoss': f'{contrastive_loss.item():.4f}',
                'ClsLoss': f'{classification_loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.3f}'
            }
            
            # 添加GPU内存使用信息
            if torch.cuda.is_available() and self.device.type == 'cuda':
                gpu_memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3
                postfix_dict['GPU_Mem'] = f'{gpu_memory_used:.1f}GB'
                postfix_dict['GPU_Cache'] = f'{gpu_memory_cached:.1f}GB'
            
            pbar.set_postfix(postfix_dict)
        
        # 计算平均值
        avg_loss = total_loss / len(train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        # 学习率调度
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        # 记录统计信息
        stats = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_contrastive_loss': avg_contrastive_loss,
            'avg_classification_loss': avg_classification_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # 保存到历史记录
        for key, value in stats.items():
            self.train_stats[key].append(value)
        
        logger.info(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, "
            f"ConLoss={avg_contrastive_loss:.4f}, "
            f"ClsLoss={avg_classification_loss:.4f}, "
            f"Acc={accuracy:.3f}, LR={stats['learning_rate']:.6f}"
        )
        
        return stats
    
    def _compute_contrastive_loss(self, face_emb, body_emb, identity_ids):
        """
        计算对比学习损失
        
        Args:
            face_emb: 人脸特征 [B, 128]
            body_emb: 身体特征 [B, 128]
            identity_ids: 身份ID [B]
            
        Returns:
            torch.Tensor: 对比学习损失
        """
        # 方法1: 使用FaceBodySupConLoss
        if hasattr(self.supcon_loss, 'forward') and \
           len(self.supcon_loss.forward.__code__.co_varnames) > 3:
            # FaceBodySupConLoss
            loss_dict = self.supcon_loss(face_emb, body_emb, identity_ids)
            return loss_dict['total_loss']
        else:
            # 方法2: 基础SupConLoss - 拼接face和body特征
            # 将face和body作为同一身份的两个视图
            combined_features = torch.stack([face_emb, body_emb], dim=1)  # [B, 2, 128]
            return self.supcon_loss(combined_features, identity_ids)
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            dict: 验证统计信息
        """
        # 设置为评估模式
        self.face_encoder.eval()
        self.body_encoder.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_classification_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # 数据移动到设备
                face_images = batch['face'].to(self.device)
                body_images = batch['body'].to(self.device)
                labels = batch['label'].to(self.device).float()
                identity_ids = batch['identity'].to(self.device)
                
                batch_size = face_images.shape[0]
                
                # 前向传播
                face_emb = self.face_encoder(face_images)
                body_emb = self.body_encoder(body_images)
                
                # 计算损失
                contrastive_loss = self._compute_contrastive_loss(face_emb, body_emb, identity_ids)
                
                classification_logits = self.classifier(face_emb, body_emb)
                classification_loss = self.bce_loss(classification_logits.squeeze(), labels)
                
                total_batch_loss = (
                    self.contrastive_weight * contrastive_loss +
                    self.classification_weight * classification_loss
                )
                
                # 计算准确率
                predictions = torch.sigmoid(classification_logits.squeeze()) > 0.5
                correct_predictions += (predictions == labels.bool()).sum().item()
                total_samples += batch_size
                
                # 累计损失
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_classification_loss += classification_loss.item()
        
        # 计算平均值
        avg_loss = total_loss / len(val_loader)
        avg_contrastive_loss = total_contrastive_loss / len(val_loader)
        avg_classification_loss = total_classification_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        stats = {
            'val_loss': avg_loss,
            'val_contrastive_loss': avg_contrastive_loss,
            'val_classification_loss': avg_classification_loss,
            'val_accuracy': accuracy
        }
        
        logger.info(
            f"Validation: Loss={avg_loss:.4f}, "
            f"ConLoss={avg_contrastive_loss:.4f}, "
            f"ClsLoss={avg_classification_loss:.4f}, "
            f"Acc={accuracy:.3f}"
        )
        
        return stats
    
    def save_checkpoint(self, filepath, epoch, best_val_loss=None):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
            epoch: 当前epoch
            best_val_loss: 最佳验证损失
        """
        checkpoint = {
            'epoch': epoch,
            'face_encoder_state_dict': self.face_encoder.state_dict(),
            'body_encoder_state_dict': self.body_encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': dict(self.train_stats),
            'best_val_loss': best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        加载检查点
        
        Args:
            filepath: 检查点路径
            
        Returns:
            dict: 检查点信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.face_encoder.load_state_dict(checkpoint['face_encoder_state_dict'])
        self.body_encoder.load_state_dict(checkpoint['body_encoder_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_stats = defaultdict(list, checkpoint.get('train_stats', {}))
        
        logger.info(f"检查点已加载: {filepath}")
        
        return {
            'epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint.get('best_val_loss')
        }
    
    def get_training_stats(self):
        """
        获取训练统计信息
        
        Returns:
            dict: 训练统计
        """
        return dict(self.train_stats)

def create_trainer(face_encoder, body_encoder, classifier, supcon_loss, **kwargs):
    """
    创建训练器
    
    Args:
        face_encoder: 人脸编码器
        body_encoder: 身体编码器
        classifier: 分类器
        supcon_loss: 对比学习损失
        **kwargs: 其他参数
        
    Returns:
        FaceBodyTrainer: 训练器实例
    """
    return FaceBodyTrainer(
        face_encoder=face_encoder,
        body_encoder=body_encoder,
        classifier=classifier,
        supcon_loss=supcon_loss,
        **kwargs
    )

def test_trainer():
    """
    测试训练器
    """
    print("测试训练器...")
    
    # 这里只是一个简单的测试，实际使用需要真实的数据加载器
    from face_encoder import FaceEncoder
    from body_encoder import BodyEncoder
    from face_body_classifier import FaceBodyClassifier
    from supcon_loss import SupConLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    face_encoder = FaceEncoder(pretrained=False)
    body_encoder = BodyEncoder(backbone='mobilenet_v2', pretrained=False)
    classifier = FaceBodyClassifier()
    supcon_loss = SupConLoss()
    
    # 创建训练器
    trainer = create_trainer(
        face_encoder=face_encoder,
        body_encoder=body_encoder,
        classifier=classifier,
        supcon_loss=supcon_loss,
        device=device
    )
    
    # 设置优化器
    trainer.setup_optimizer(lr=1e-3)
    
    print(f"训练器创建成功，使用设备: {device}")
    print(f"优化器参数组数: {len(trainer.optimizer.param_groups)}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in trainer.face_encoder.parameters()) + \
                  sum(p.numel() for p in trainer.body_encoder.parameters()) + \
                  sum(p.numel() for p in trainer.classifier.parameters())
    
    print(f"总参数量: {total_params:,}")
    
    print("训练器测试完成！")

if __name__ == "__main__":
    test_trainer()