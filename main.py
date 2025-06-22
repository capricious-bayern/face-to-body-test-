#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主训练脚本
人脸-身体伪造检测模型训练
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
import json
from datetime import datetime
import numpy as np
from pathlib import Path

# 导入自定义模块
from data_preprocessing import VideoProcessor
from face_encoder import create_face_encoder
from body_encoder import create_body_encoder
from face_body_classifier import create_classifier
from supcon_loss import SupConLoss, FaceBodySupConLoss
from trainer import create_trainer
from dataset import FaceBodyDataset, VideoFaceBodyDataset, create_dataloader, create_mock_dataset

# 导入新的分批数据集
from batched_dataset import BatchedDataset, create_batched_dataloader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='人脸-身体伪造检测模型训练')
    
    # 数据相关参数
    parser.add_argument('--dataset_path', type=str, default=r'D:\Dataset\Celeb-DF-v2',
                       help='数据集路径')
    parser.add_argument('--use_mock_data', action='store_true',
                       help='使用模拟数据进行测试')
    parser.add_argument('--mock_samples', type=int, default=200,
                       help='模拟数据样本数量')
    parser.add_argument('--mock_identities', type=int, default=20,
                       help='模拟数据身份数量')
    parser.add_argument('--data_file', type=str, default=None,
                       help='处理后的Celeb-DF数据文件路径')
    parser.add_argument('--prepare_data', action='store_true', default=False,
                       help='是否先处理原始视频数据')
    parser.add_argument('--use_batched_dataset', action='store_true', default=True,
                       help='是否使用分批数据集（内存高效）')
    parser.add_argument('--index_file', type=str, default=None,
                       help='数据索引文件路径（分批数据集模式）')
    parser.add_argument('--frame_interval', type=int, default=30,
                       help='视频帧间隔')
    parser.add_argument('--max_videos', type=int, default=10,
                       help='最大处理视频数量')
    parser.add_argument('--max_frames_per_video', type=int, default=5,
                       help='每个视频最大帧数')
    
    # 模型相关参数
    parser.add_argument('--face_encoder_type', type=str, default='basic',
                       choices=['basic', 'attention'],
                       help='人脸编码器类型')
    parser.add_argument('--body_encoder_type', type=str, default='basic',
                       choices=['basic', 'multiscale', 'bodypart'],
                       help='身体编码器类型')
    parser.add_argument('--classifier_type', type=str, default='basic',
                       choices=['basic', 'advanced', 'metric'],
                       help='分类器类型')
    parser.add_argument('--face_backbone', type=str, default='mobilenet_v2', help='Backbone for face encoder (e.g., mobilenet_v2)')
    parser.add_argument('--body_backbone', type=str, default='efficientnet-b0', help='Backbone for body encoder (e.g., efficientnet-b0)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='学习率调度器')
    
    # 损失函数参数
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='对比学习温度系数')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                       help='对比学习损失权重')
    parser.add_argument('--classification_weight', type=float, default=0.5,
                       help='分类损失权重')
    parser.add_argument('--negative_ratio', type=float, default=0.5,
                       help='负样本比例')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='模型保存间隔')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='使用预训练权重')
    
    return parser.parse_args()

def setup_device(device_arg):
    """
    设置计算设备
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"使用计算设备: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU信息: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_models(args, device):
    """
    创建模型
    """
    logger.info("创建模型...")
    
    # 创建人脸编码器
    face_encoder = create_face_encoder(
        encoder_type=args.face_encoder_type,
        pretrained=args.pretrained
    )
    
    # 创建身体编码器
    body_encoder = create_body_encoder(
        encoder_type=args.body_encoder_type,
        backbone=args.body_backbone,
        pretrained=args.pretrained
    )
    
    # 创建分类器
    classifier = create_classifier(
        classifier_type=args.classifier_type
    )
    
    # 创建对比学习损失
    if args.face_encoder_type == 'basic' and args.body_encoder_type == 'basic':
        supcon_loss = FaceBodySupConLoss(temperature=args.temperature)
    else:
        supcon_loss = SupConLoss(temperature=args.temperature)
    
    # 移动到设备
    face_encoder = face_encoder.to(device)
    body_encoder = body_encoder.to(device)
    classifier = classifier.to(device)
    
    # 打印模型信息
    face_params = sum(p.numel() for p in face_encoder.parameters())
    body_params = sum(p.numel() for p in body_encoder.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    total_params = face_params + body_params + classifier_params
    
    logger.info(f"人脸编码器参数量: {face_params:,}")
    logger.info(f"身体编码器参数量: {body_params:,}")
    logger.info(f"分类器参数量: {classifier_params:,}")
    logger.info(f"总参数量: {total_params:,}")
    
    return face_encoder, body_encoder, classifier, supcon_loss

def load_data(args):
    """
    加载数据集
    """
    if args.prepare_data:
        print("开始处理原始视频数据...")
        from prepare_celeb_df_data import prepare_data
        data_file = prepare_data()
        if data_file:
            args.data_file = data_file
            print(f"数据处理完成，保存到: {data_file}")
        else:
            print("数据处理失败，使用模拟数据")
            args.use_mock_data = True
    
    if args.data_file and os.path.exists(args.data_file):
        print(f"加载处理后的Celeb-DF数据: {args.data_file}")
        from prepare_celeb_df_data import load_processed_data
        
        try:
            data_dict = load_processed_data(args.data_file)
            face_tensors = data_dict['face_tensors']
            body_tensors = data_dict['body_tensors']
            identity_labels = data_dict['identities']
            
            print(f"成功加载 {len(face_tensors)} 个样本")
            print(f"- 真实样本: {data_dict['num_real']}")
            print(f"- 合成样本: {data_dict['num_fake']}")
            print(f"- 身份数量: {data_dict['num_identities']}")
            
        except Exception as e:
            print(f"加载数据文件失败: {e}")
            print("使用模拟数据")
            args.use_mock_data = True
    
    if args.use_mock_data:
        print("使用模拟数据进行训练...")
        # 生成模拟数据
        face_tensors = []
        body_tensors = []
        identity_labels = []
        
        for i in range(args.mock_samples):
            # 生成随机图像张量
            face_tensor = torch.randn(3, 256, 256)
            body_tensor = torch.randn(3, 256, 256)
            identity = i % args.mock_identities  # 循环分配身份
            
            face_tensors.append(face_tensor)
            body_tensors.append(body_tensor)
            identity_labels.append(identity)
        
        print(f"生成了 {len(face_tensors)} 个模拟样本，{args.mock_identities} 个身份")
    
    # 创建数据集
    dataset = FaceBodyDataset(
        face_tensors=face_tensors,
        body_tensors=body_tensors,
        identity_labels=identity_labels,
        augment=True,
        negative_ratio=0.5
    )
    
    return dataset

def create_dataset(args, device):
    """
    创建数据集
    """
    if args.use_batched_dataset:
        # 使用分批数据集，不需要在这里划分，直接返回None
        # 分批数据集的划分在数据加载器中处理
        return None, None
    else:
        # 传统数据集模式
        dataset = load_data(args)
        
        # 划分训练和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset

def main():
    """
    主训练函数
    """
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"训练配置已保存: {config_path}")
    logger.info(f"训练参数: {vars(args)}")
    
    # 2. Device configuration
    # 优先使用CUDA设备，如果不可用则回退到CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 设备信息 ===")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA可用: True")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU内存已用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    else:
        print(f"CUDA可用: False")
        print(f"将使用CPU进行训练")
    print(f"==================\n")
    
    # 创建模型
    face_encoder, body_encoder, classifier, supcon_loss = create_models(args, device)
    
    # 准备数据
    if args.prepare_data:
        print("\n准备数据集...")
        from prepare_celeb_df_data import prepare_data
        data_file = prepare_data(args.dataset_path)
        if data_file:
            print(f"数据准备完成，文件保存在: {data_file}")
            if args.use_batched_dataset:
                args.index_file = data_file
        else:
            print("数据准备失败")
            return
    
    # 创建数据加载器
    if args.use_batched_dataset:
        # 使用分批数据集（内存高效）
        if not args.index_file:
            # 尝试查找最新的索引文件
            processed_dir = "processed_data"
            if os.path.exists(processed_dir):
                index_files = [f for f in os.listdir(processed_dir) if f.startswith("celeb_df_index_")]
                if index_files:
                    # 按文件名排序（包含时间戳）
                    index_files.sort(reverse=True)
                    args.index_file = os.path.join(processed_dir, index_files[0])
                    print(f"自动选择最新的索引文件: {args.index_file}")
        
        if not args.index_file or not os.path.exists(args.index_file):
            print("错误: 未指定有效的索引文件，请先准备数据或指定正确的索引文件路径")
            return
        
        print(f"\n使用分批数据集加载数据: {args.index_file}")
        train_loader = create_batched_dataloader(
            index_file=args.index_file,
            batch_size=args.batch_size,
            negative_ratio=args.negative_ratio,
            augment=True,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        # 获取数据集统计信息
        dataset = train_loader.dataset
        stats = dataset.get_statistics()
        print(f"数据集统计:")
        print(f"- 总样本数: {stats['total_samples']}")
        print(f"- 真实样本: {stats['num_real']}")
        print(f"- 合成样本: {stats['num_fake']}")
        print(f"- 身份数量: {stats['num_identities']}")
        
        # 分批数据集模式下，验证集使用相同的数据加载器
        val_loader = create_batched_dataloader(
            index_file=args.index_file,
            batch_size=args.batch_size,
            negative_ratio=args.negative_ratio,
            augment=False,  # 验证时不使用数据增强
            shuffle=False,
            num_workers=args.num_workers
        )
        
    else:
        # 传统数据集模式
        train_dataset, val_dataset = create_dataset(args, device)
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    if args.use_batched_dataset:
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
    else:
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"验证批次数: {len(val_loader)}")
    
    # 创建训练器
    trainer = create_trainer(
        face_encoder=face_encoder,
        body_encoder=body_encoder,
        classifier=classifier,
        supcon_loss=supcon_loss,
        device=device,
        contrastive_weight=args.contrastive_weight,
        classification_weight=args.classification_weight
    )
    
    # 设置优化器
    trainer.setup_optimizer(
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler
    )
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint_info = trainer.load_checkpoint(args.resume)
            start_epoch = checkpoint_info['epoch'] + 1
            best_val_loss = checkpoint_info.get('best_val_loss', float('inf'))
            logger.info(f"从epoch {start_epoch}恢复训练")
        else:
            logger.warning(f"检查点文件不存在: {args.resume}")
    
    # 训练循环
    logger.info("开始训练...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # 训练一个epoch
        train_stats = trainer.train_one_epoch(train_loader, epoch + 1)
        
        # 验证
        val_stats = trainer.validate(val_loader)
        
        # 保存最佳模型
        val_loss = val_stats['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_dir / 'best_model.pth'
            trainer.save_checkpoint(best_model_path, epoch + 1, best_val_loss)
            logger.info(f"保存最佳模型: {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch + 1, best_val_loss)
        
        # 保存训练统计
        stats_path = save_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(trainer.get_training_stats(), f, indent=2)
    
    # 保存最终模型
    final_model_path = save_dir / 'final_model.pth'
    trainer.save_checkpoint(final_model_path, args.epochs, best_val_loss)
    
    logger.info(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"模型已保存到: {save_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise