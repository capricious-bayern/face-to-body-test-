import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional

# 导入自定义模块
from models.face_encoder import create_face_encoder
from models.body_encoder import create_body_encoder
from models.contrastive_loss import create_contrastive_loss
from models.classifier import create_classifier
from datasets.video_dataset import create_dataloader
from utils.metrics import calculate_metrics, AverageMeter
from utils.checkpoint import save_checkpoint, load_checkpoint

class FaceBodyTrainer:
    """
    人脸-身体匹配检测训练器
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda')
        
        # 设置日志
        self._setup_logging()
        
        # 创建模型
        self._create_models()
        
        # 创建优化器和调度器
        self._create_optimizers()
        
        # 创建损失函数
        self._create_loss_functions()
        
        # 创建数据加载器
        self._create_dataloaders()
        
        # 设置tensorboard
        self._setup_tensorboard()
        
        # 训练状态
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.global_step = 0
        
    def _setup_logging(self):
        """设置日志"""
        log_dir = self.config['training']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_models(self):
        """创建模型"""
        # 人脸编码器
        self.face_encoder = create_face_encoder(
            model_type=self.config['model']['face_encoder']['type'],
            **self.config['model']['face_encoder']['params']
        ).to(self.device)
        
        # 身体编码器
        self.body_encoder = create_body_encoder(
            model_type=self.config['model']['body_encoder']['type'],
            **self.config['model']['body_encoder']['params']
        ).to(self.device)
        
        # 分类器
        self.classifier = create_classifier(
            classifier_type=self.config['model']['classifier']['type'],
            **self.config['model']['classifier']['params']
        ).to(self.device)
        
        self.logger.info(f"Models created and moved to {self.device}")
        
    def _create_optimizers(self):
        """创建优化器和学习率调度器"""
        # 分别为编码器和分类器设置不同的学习率
        encoder_params = list(self.face_encoder.parameters()) + list(self.body_encoder.parameters())
        classifier_params = list(self.classifier.parameters())
        
        self.encoder_optimizer = optim.AdamW(
            encoder_params,
            lr=self.config['training']['encoder_lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.classifier_optimizer = optim.AdamW(
            classifier_params,
            lr=self.config['training']['classifier_lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['min_lr']
        )
        
        self.classifier_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.classifier_optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['min_lr']
        )
        
    def _create_loss_functions(self):
        """创建损失函数"""
        # 对比学习损失
        self.contrastive_loss = create_contrastive_loss(
            loss_type=self.config['loss']['contrastive']['type'],
            **self.config['loss']['contrastive']['params']
        )
        
        # 分类损失
        self.classification_loss = nn.CrossEntropyLoss(
            label_smoothing=self.config['loss']['classification'].get('label_smoothing', 0.0)
        )
        
        # 损失权重
        self.contrastive_weight = self.config['loss']['weights']['contrastive']
        self.classification_weight = self.config['loss']['weights']['classification']
        
    def _create_dataloaders(self):
        """创建数据加载器"""
        # 训练数据加载器
        self.train_loader = create_dataloader(
            dataset_path=self.config['data']['train_path'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            shuffle=True,
            contrastive=True  # 用于对比学习
        )
        
        # 验证数据加载器
        self.val_loader = create_dataloader(
            dataset_path=self.config['data']['val_path'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            shuffle=False,
            contrastive=False  # 用于分类评估
        )
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        
    def _setup_tensorboard(self):
        """设置tensorboard"""
        log_dir = os.path.join(self.config['training']['log_dir'], 'tensorboard')
        self.writer = SummaryWriter(log_dir)
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.face_encoder.train()
        self.body_encoder.train()
        self.classifier.train()
        
        # 损失记录器
        contrastive_meter = AverageMeter()
        classification_meter = AverageMeter()
        total_meter = AverageMeter()
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 解包数据
            if len(batch_data) == 6:  # 对比学习数据
                anchor_face, anchor_body, pos_face, pos_body, neg_face, neg_body = batch_data
                anchor_face = anchor_face.to(self.device)
                anchor_body = anchor_body.to(self.device)
                pos_face = pos_face.to(self.device)
                pos_body = pos_body.to(self.device)
                neg_face = neg_face.to(self.device)
                neg_body = neg_body.to(self.device)
                
                # 提取特征
                anchor_face_feat = self.face_encoder.get_projections(anchor_face)
                anchor_body_feat = self.body_encoder.get_projections(anchor_body)
                pos_face_feat = self.face_encoder.get_projections(pos_face)
                pos_body_feat = self.body_encoder.get_projections(pos_body)
                neg_face_feat = self.face_encoder.get_projections(neg_face)
                neg_body_feat = self.body_encoder.get_projections(neg_body)
                
                # 对比学习损失
                contrastive_loss = self.contrastive_loss(anchor_face_feat, anchor_body_feat)
                
                # 分类损失（使用正负样本）
                # 正样本
                pos_face_cls_feat = self.face_encoder.get_features(anchor_face)
                pos_body_cls_feat = self.body_encoder.get_features(anchor_body)
                pos_logits = self.classifier(pos_face_cls_feat, pos_body_cls_feat)
                pos_labels = torch.ones(pos_logits.size(0), dtype=torch.long).to(self.device)
                
                # 负样本
                neg_face_cls_feat = self.face_encoder.get_features(anchor_face)
                neg_body_cls_feat = self.body_encoder.get_features(neg_body)
                neg_logits = self.classifier(neg_face_cls_feat, neg_body_cls_feat)
                neg_labels = torch.zeros(neg_logits.size(0), dtype=torch.long).to(self.device)
                
                # 合并正负样本
                all_logits = torch.cat([pos_logits, neg_logits], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)
                
                classification_loss = self.classification_loss(all_logits, all_labels)
                
            else:  # 普通分类数据
                face_imgs, body_imgs, labels = batch_data
                face_imgs = face_imgs.to(self.device)
                body_imgs = body_imgs.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                face_features = self.face_encoder.get_features(face_imgs)
                body_features = self.body_encoder.get_features(body_imgs)
                
                # 分类
                logits = self.classifier(face_features, body_features)
                classification_loss = self.classification_loss(logits, labels)
                
                # 对比学习损失（简化版）
                face_proj = self.face_encoder.get_projections(face_imgs)
                body_proj = self.body_encoder.get_projections(body_imgs)
                contrastive_loss = self.contrastive_loss(face_proj, body_proj)
            
            # 总损失
            total_loss = (self.contrastive_weight * contrastive_loss + 
                         self.classification_weight * classification_loss)
            
            # 反向传播
            self.encoder_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.face_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.body_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            
            self.encoder_optimizer.step()
            self.classifier_optimizer.step()
            
            # 记录损失
            batch_size = anchor_face.size(0) if len(batch_data) == 6 else face_imgs.size(0)
            contrastive_meter.update(contrastive_loss.item(), batch_size)
            classification_meter.update(classification_loss.item(), batch_size)
            total_meter.update(total_loss.item(), batch_size)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Contrastive': f'{contrastive_meter.avg:.4f}',
                'Classification': f'{classification_meter.avg:.4f}',
                'Total': f'{total_meter.avg:.4f}'
            })
            
            # 记录到tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/ContrastiveLoss', contrastive_loss.item(), self.global_step)
                self.writer.add_scalar('Train/ClassificationLoss', classification_loss.item(), self.global_step)
                self.writer.add_scalar('Train/TotalLoss', total_loss.item(), self.global_step)
            
            self.global_step += 1
        
        return {
            'contrastive_loss': contrastive_meter.avg,
            'classification_loss': classification_meter.avg,
            'total_loss': total_meter.avg
        }
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.face_encoder.eval()
        self.body_encoder.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc='Validation'):
                face_imgs, body_imgs, labels = batch_data
                face_imgs = face_imgs.to(self.device)
                body_imgs = body_imgs.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                face_features = self.face_encoder.get_features(face_imgs)
                body_features = self.body_encoder.get_features(body_imgs)
                
                # 分类
                logits = self.classifier(face_features, body_features)
                loss = self.classification_loss(logits, labels)
                
                # 预测
                predictions = torch.argmax(logits, dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loss.update(loss.item(), face_imgs.size(0))
        
        # 计算指标
        metrics = calculate_metrics(all_labels, all_predictions)
        metrics['val_loss'] = val_loss.avg
        
        return metrics
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 更新学习率
            self.encoder_scheduler.step()
            self.classifier_scheduler.step()
            
            # 记录指标
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['total_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Tensorboard记录
            self.writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Epoch/ValAccuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Epoch/ValF1', val_metrics['f1'], epoch)
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.save_checkpoint(is_best=True)
                self.logger.info(f"New best accuracy: {self.best_accuracy:.4f}")
            
            # 定期保存检查点
            if epoch % self.config['training']['save_freq'] == 0:
                self.save_checkpoint(is_best=False)
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'face_encoder_state_dict': self.face_encoder.state_dict(),
            'body_encoder_state_dict': self.body_encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict(),
            'encoder_scheduler_state_dict': self.encoder_scheduler.state_dict(),
            'classifier_scheduler_state_dict': self.classifier_scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }
        
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            save_path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Face-Body Matching Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = FaceBodyTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()