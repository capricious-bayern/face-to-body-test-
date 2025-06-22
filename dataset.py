#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集类
用于加载和处理人脸-身体伪造检测数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class FaceBodyDataset(Dataset):
    """
    人脸-身体数据集
    
    数据格式:
    - face_tensors: 人脸图像张量列表
    - body_tensors: 身体图像张量列表
    - identity_labels: 身份标签列表
    """
    
    def __init__(self, 
                 face_tensors, 
                 body_tensors, 
                 identity_labels,
                 augment=True,
                 negative_ratio=0.5):
        """
        初始化数据集
        
        Args:
            face_tensors: 人脸张量列表 [N, 3, 256, 256]
            body_tensors: 身体张量列表 [N, 3, 256, 256]
            identity_labels: 身份标签列表 [N]
            augment: 是否进行数据增强
            negative_ratio: 负样本比例（伪造样本）
        """
        self.face_tensors = face_tensors
        self.body_tensors = body_tensors
        self.identity_labels = identity_labels
        self.augment = augment
        self.negative_ratio = negative_ratio
        
        # 验证数据一致性
        assert len(face_tensors) == len(body_tensors) == len(identity_labels), \
            "人脸、身体张量和标签数量必须一致"
        
        # 按身份组织数据
        self.identity_to_indices = defaultdict(list)
        for idx, identity in enumerate(identity_labels):
            self.identity_to_indices[identity].append(idx)
        
        self.unique_identities = list(self.identity_to_indices.keys())
        
        # 生成样本对
        self._generate_pairs()
        
        logger.info(f"数据集初始化完成: {len(self.pairs)} 个样本对")
        logger.info(f"正样本: {self.positive_count}, 负样本: {self.negative_count}")
    
    def _generate_pairs(self):
        """
        生成正负样本对
        """
        self.pairs = []
        
        # 生成正样本对（同一身份的face-body配对）
        positive_pairs = []
        for identity, indices in self.identity_to_indices.items():
            for idx in indices:
                positive_pairs.append((idx, idx, 1, identity))  # (face_idx, body_idx, label, identity)
        
        # 生成负样本对（不同身份的face-body配对）
        negative_pairs = []
        num_negatives = int(len(positive_pairs) * self.negative_ratio / (1 - self.negative_ratio))
        
        for _ in range(num_negatives):
            # 随机选择两个不同身份
            identity1, identity2 = random.sample(self.unique_identities, 2)
            
            # 随机选择每个身份的一个样本
            face_idx = random.choice(self.identity_to_indices[identity1])
            body_idx = random.choice(self.identity_to_indices[identity2])
            
            negative_pairs.append((face_idx, body_idx, 0, identity1))  # 使用face的身份作为identity
        
        # 合并正负样本
        self.pairs = positive_pairs + negative_pairs
        
        # 打乱顺序
        random.shuffle(self.pairs)
        
        self.positive_count = len(positive_pairs)
        self.negative_count = len(negative_pairs)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            dict: {
                'face': torch.Tensor [3, 256, 256],
                'body': torch.Tensor [3, 256, 256],
                'label': int (0=伪造, 1=真实),
                'identity': int (身份ID)
            }
        """
        face_idx, body_idx, label, identity = self.pairs[idx]
        
        # 获取人脸和身体图像
        face_tensor = self.face_tensors[face_idx].clone()
        body_tensor = self.body_tensors[body_idx].clone()
        
        # 数据增强
        if self.augment:
            face_tensor = self._augment_tensor(face_tensor)
            body_tensor = self._augment_tensor(body_tensor)
        
        return {
            'face': face_tensor,
            'body': body_tensor,
            'label': label,
            'identity': identity
        }
    
    def _augment_tensor(self, tensor):
        """
        对张量进行数据增强
        
        Args:
            tensor: 输入张量 [3, 256, 256]
            
        Returns:
            torch.Tensor: 增强后的张量
        """
        # 随机水平翻转
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[2])
        
        # 随机亮度调整
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            tensor = torch.clamp(tensor * brightness_factor, 0, 1)
        
        # 随机对比度调整
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = tensor.mean()
            tensor = torch.clamp((tensor - mean) * contrast_factor + mean, 0, 1)
        
        # 随机噪声
        if random.random() > 0.8:
            noise = torch.randn_like(tensor) * 0.01
            tensor = torch.clamp(tensor + noise, 0, 1)
        
        return tensor
    
    def get_class_weights(self):
        """
        计算类别权重，用于处理类别不平衡
        
        Returns:
            torch.Tensor: 类别权重 [negative_weight, positive_weight]
        """
        total_samples = len(self.pairs)
        positive_weight = total_samples / (2 * self.positive_count)
        negative_weight = total_samples / (2 * self.negative_count)
        
        return torch.tensor([negative_weight, positive_weight])

class VideoFaceBodyDataset(Dataset):
    """
    基于视频的人脸-身体数据集
    直接从视频处理器获取数据
    """
    
    def __init__(self, 
                 video_processor,
                 video_paths,
                 identity_mapping,
                 frame_interval=30,
                 max_frames_per_video=10,
                 negative_ratio=0.5,
                 augment=True):
        """
        初始化视频数据集
        
        Args:
            video_processor: 视频处理器实例
            video_paths: 视频路径列表
            identity_mapping: 视频路径到身份ID的映射
            frame_interval: 帧间隔
            max_frames_per_video: 每个视频最大帧数
            negative_ratio: 负样本比例
            augment: 是否数据增强
        """
        self.video_processor = video_processor
        self.video_paths = video_paths
        self.identity_mapping = identity_mapping
        self.frame_interval = frame_interval
        self.max_frames_per_video = max_frames_per_video
        self.negative_ratio = negative_ratio
        self.augment = augment
        
        # 处理所有视频，提取人脸和身体特征
        self._process_videos()
        
        # 生成训练样本对
        self._generate_pairs()
    
    def _process_videos(self):
        """
        处理所有视频，提取人脸和身体图像
        """
        self.face_tensors = []
        self.body_tensors = []
        self.identity_labels = []
        
        logger.info(f"开始处理 {len(self.video_paths)} 个视频...")
        
        for video_path in self.video_paths:
            identity = self.identity_mapping.get(video_path, 0)
            
            # 处理视频
            face_tensors, body_tensors = self.video_processor.process_video(
                video_path, 
                self.frame_interval, 
                self.max_frames_per_video
            )
            
            # 添加到数据集
            self.face_tensors.extend(face_tensors)
            self.body_tensors.extend(body_tensors)
            self.identity_labels.extend([identity] * len(face_tensors))
        
        logger.info(f"视频处理完成，共提取 {len(self.face_tensors)} 对图像")
    
    def _generate_pairs(self):
        """
        生成正负样本对
        """
        # 按身份组织数据
        identity_to_indices = defaultdict(list)
        for idx, identity in enumerate(self.identity_labels):
            identity_to_indices[identity].append(idx)
        
        unique_identities = list(identity_to_indices.keys())
        
        self.pairs = []
        
        # 生成正样本对
        positive_pairs = []
        for identity, indices in identity_to_indices.items():
            for idx in indices:
                positive_pairs.append((idx, idx, 1, identity))
        
        # 生成负样本对
        negative_pairs = []
        num_negatives = int(len(positive_pairs) * self.negative_ratio / (1 - self.negative_ratio))
        
        for _ in range(num_negatives):
            identity1, identity2 = random.sample(unique_identities, 2)
            face_idx = random.choice(identity_to_indices[identity1])
            body_idx = random.choice(identity_to_indices[identity2])
            negative_pairs.append((face_idx, body_idx, 0, identity1))
        
        self.pairs = positive_pairs + negative_pairs
        random.shuffle(self.pairs)
        
        logger.info(f"生成样本对完成: 正样本 {len(positive_pairs)}, 负样本 {len(negative_pairs)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        face_idx, body_idx, label, identity = self.pairs[idx]
        
        face_tensor = self.face_tensors[face_idx].clone()
        body_tensor = self.body_tensors[body_idx].clone()
        
        if self.augment:
            face_tensor = self._augment_tensor(face_tensor)
            body_tensor = self._augment_tensor(body_tensor)
        
        return {
            'face': face_tensor,
            'body': body_tensor,
            'label': label,
            'identity': identity
        }
    
    def _augment_tensor(self, tensor):
        """数据增强（与FaceBodyDataset相同）"""
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[2])
        
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            tensor = torch.clamp(tensor * brightness_factor, 0, 1)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = tensor.mean()
            tensor = torch.clamp((tensor - mean) * contrast_factor + mean, 0, 1)
        
        if random.random() > 0.8:
            noise = torch.randn_like(tensor) * 0.01
            tensor = torch.clamp(tensor + noise, 0, 1)
        
        return tensor

def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4, **kwargs):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **kwargs: 其他参数
        
    Returns:
        DataLoader: 数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )

def create_mock_dataset(num_samples=100, num_identities=10):
    """
    创建模拟数据集用于测试
    
    Args:
        num_samples: 样本数量
        num_identities: 身份数量
        
    Returns:
        FaceBodyDataset: 模拟数据集
    """
    # 生成随机人脸和身体张量
    face_tensors = []
    body_tensors = []
    identity_labels = []
    
    for i in range(num_samples):
        # 随机生成图像张量 [3, 256, 256]
        face_tensor = torch.rand(3, 256, 256)
        body_tensor = torch.rand(3, 256, 256)
        
        # 随机分配身份
        identity = i % num_identities
        
        face_tensors.append(face_tensor)
        body_tensors.append(body_tensor)
        identity_labels.append(identity)
    
    return FaceBodyDataset(
        face_tensors=face_tensors,
        body_tensors=body_tensors,
        identity_labels=identity_labels,
        augment=True,
        negative_ratio=0.5
    )

def test_dataset():
    """
    测试数据集
    """
    print("测试数据集...")
    
    # 创建模拟数据集
    dataset = create_mock_dataset(num_samples=50, num_identities=5)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"正样本数量: {dataset.positive_count}")
    print(f"负样本数量: {dataset.negative_count}")
    
    # 测试数据加载
    dataloader = create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    print(f"数据加载器批次数: {len(dataloader)}")
    
    # 测试一个批次
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n批次 {batch_idx}:")
        print(f"Face形状: {batch['face'].shape}")
        print(f"Body形状: {batch['body'].shape}")
        print(f"Label: {batch['label']}")
        print(f"Identity: {batch['identity']}")
        
        # 验证数据范围
        print(f"Face数值范围: [{batch['face'].min():.3f}, {batch['face'].max():.3f}]")
        print(f"Body数值范围: [{batch['body'].min():.3f}, {batch['body'].max():.3f}]")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    # 测试类别权重
    class_weights = dataset.get_class_weights()
    print(f"\n类别权重: {class_weights}")
    
    print("数据集测试完成！")

if __name__ == "__main__":
    test_dataset()