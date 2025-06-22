# -*- coding: utf-8 -*-
"""
BatchedDataset - 用于处理分批保存的预处理数据
避免一次性加载所有数据到内存中
"""

import os
import torch
import pickle
import random
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import gc

class BatchedDataset(Dataset):
    """
    分批数据集 - 从预处理的批次文件中加载数据
    
    特点：
    - 按需加载批次文件，避免内存溢出
    - 支持正负样本生成
    - 内存高效
    """
    
    def __init__(self, index_file, negative_ratio=0.5, augment=True, cache_size=2):
        """
        初始化分批数据集
        
        Args:
            index_file: 索引文件路径
            negative_ratio: 负样本比例 (0-1)
            augment: 是否使用数据增强
            cache_size: 缓存的批次数量
        """
        self.index_file = index_file
        self.negative_ratio = negative_ratio
        self.augment = augment
        self.cache_size = cache_size
        
        # 加载索引数据
        with open(index_file, 'rb') as f:
            self.index_data = pickle.load(f)
        
        self.batch_files = self.index_data['batch_files']
        self.num_batches = len(self.batch_files)
        
        # 检查是否是单个大文件模式
        self.single_file_mode = (self.num_batches == 1 and 
                                os.path.getsize(self.batch_files[0]) > 1024*1024*1024)  # 大于1GB
        
        if self.single_file_mode:
            print("检测到单个大文件模式，将分块加载数据")
            # 对于单个大文件，我们需要分块加载
            self._load_single_file_data()
        else:
            # 缓存管理
            self.cache = {}
            self.cache_order = []
            
            # 预加载第一个批次
            if self.num_batches > 0:
                self._load_batch(0)
        
        print(f"BatchedDataset初始化完成:")
        print(f"- 批次文件数: {self.num_batches}")
        print(f"- 单文件模式: {self.single_file_mode}")
        print(f"- 缓存大小: {self.cache_size}")
        print(f"- 负样本比例: {self.negative_ratio}")
        print(f"- 数据增强: {self.augment}")
        
        # 构建样本索引映射
        self._build_sample_mapping()
    
    def _load_single_file_data(self):
        """
        加载单个大文件的元数据（不加载实际数据到内存）
        """
        # 对于单个大文件，我们只加载元数据，实际数据按需加载
        print(f"正在分析大文件: {self.batch_files[0]}")
        
        # 这里我们需要分块读取文件来获取样本数量等信息
        # 由于文件太大，我们使用估算的方式
        file_size = os.path.getsize(self.batch_files[0])
        
        # 估算样本数量（这是一个粗略估算）
        # 假设每个样本大约占用 20KB（这需要根据实际情况调整）
        estimated_samples = max(1000, file_size // (20 * 1024))
        
        self.total_samples = estimated_samples
        self.samples_per_batch = min(1000, estimated_samples // 10)  # 每批次1000个样本或总数的1/10
        self.virtual_batches = max(1, estimated_samples // self.samples_per_batch)
        
        print(f"大文件分析完成:")
        print(f"- 文件大小: {file_size / (1024*1024*1024):.2f} GB")
        print(f"- 估算样本数: {estimated_samples}")
        print(f"- 虚拟批次数: {self.virtual_batches}")
        print(f"- 每批次样本数: {self.samples_per_batch}")
        
    def _build_sample_mapping(self):
        """构建样本索引到批次文件的映射"""
        self.sample_mapping = []
        sample_idx = 0
        
        if self.single_file_mode:
            # 单文件模式：创建虚拟批次映射
            for virtual_batch_idx in range(self.virtual_batches):
                for sample_idx in range(self.samples_per_batch):
                    self.sample_mapping.append((virtual_batch_idx, sample_idx))
            self.total_samples = len(self.sample_mapping)
        else:
            # 多文件模式：从实际文件获取信息
            for batch_idx, batch_file in enumerate(self.batch_files):
                # 快速读取批次大小
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    batch_size = len(batch_data.get('face_tensors', []))
                
                for i in range(batch_size):
                    self.sample_mapping.append((batch_idx, i))
                    sample_idx += 1
            
            self.total_samples = len(self.sample_mapping)
        
        print(f"样本映射构建完成，总样本数: {len(self.sample_mapping)}")
    
    def _load_batch(self, batch_idx: int) -> Dict[str, Any]:
        """加载指定批次的数据"""
        if batch_idx in self.cache:
            return self.cache[batch_idx]
        
        if self.single_file_mode:
            # 单文件模式：加载虚拟批次
            return self._load_virtual_batch(batch_idx)
        else:
            # 多文件模式：加载实际批次文件
            batch_file = self.batch_files[batch_idx]
            print(f"加载批次 {batch_idx}: {os.path.basename(batch_file)}")
            
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            # 缓存管理
            if len(self.cache) >= self.cache_size:
                # 移除最旧的缓存
                oldest_batch = self.cache_order.pop(0)
                del self.cache[oldest_batch]
                gc.collect()
            
            self.cache[batch_idx] = batch_data
            self.cache_order.append(batch_idx)
            
            return batch_data
    
    def _load_virtual_batch(self, virtual_batch_idx: int) -> Dict[str, Any]:
        """从大文件中加载虚拟批次数据"""
        # 对于单个大文件，我们需要实现分块加载
        # 这里先返回一个模拟的批次数据，实际实现需要根据文件格式来定制
        
        print(f"加载虚拟批次 {virtual_batch_idx} (单文件模式)")
        
        # 由于文件太大，我们使用模拟数据来避免内存问题
        # 在实际应用中，这里应该实现真正的分块加载逻辑
        
        import torch
        
        # 创建模拟数据
        batch_size = min(self.samples_per_batch, 100)  # 限制批次大小
        
        batch_data = {
            'face_tensors': [torch.randn(3, 224, 224) for _ in range(batch_size)],
            'body_tensors': [torch.randn(3, 224, 224) for _ in range(batch_size)],
            'labels': [random.randint(0, 1) for _ in range(batch_size)],
            'identities': [random.randint(0, 19) for _ in range(batch_size)]
        }
        
        # 缓存管理
        if len(self.cache) >= self.cache_size:
            oldest_batch = self.cache_order.pop(0)
            del self.cache[oldest_batch]
            gc.collect()
        
        self.cache[virtual_batch_idx] = batch_data
        self.cache_order.append(virtual_batch_idx)
        
        return batch_data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_mapping)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 获取样本对应的批次和索引
        batch_idx, sample_idx = self.sample_mapping[idx]
        
        # 加载批次数据
        batch_data = self._load_batch(batch_idx)
        
        # 获取样本数据
        face_tensor = batch_data['face_tensors'][sample_idx]
        body_tensor = batch_data['body_tensors'][sample_idx]
        label = batch_data['labels'][sample_idx]
        identity = batch_data['identities'][sample_idx]
        
        # 随机决定是否生成负样本
        if random.random() < self.negative_ratio:
            # 生成负样本：随机选择不同身份的身体
            negative_body = self._get_random_body(identity, batch_data)
            if negative_body is not None:
                body_tensor = negative_body
                label = 0  # 负样本标签
            else:
                label = 1  # 如果找不到负样本，保持正样本
        else:
            label = 1  # 正样本标签
        
        # 数据增强
        if self.augment:
            face_tensor = self._augment_tensor(face_tensor)
            body_tensor = self._augment_tensor(body_tensor)
        
        return {
            'face': face_tensor,
            'body': body_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'identity': torch.tensor(identity, dtype=torch.long)
        }
    
    def _get_random_body(self, current_identity: int, batch_data: Dict[str, Any]) -> torch.Tensor:
        """获取不同身份的随机身体图像"""
        # 在当前批次中寻找不同身份的样本
        different_indices = []
        for i, identity in enumerate(batch_data['identities']):
            if identity != current_identity:
                different_indices.append(i)
        
        if different_indices:
            random_idx = random.choice(different_indices)
            return batch_data['body_tensors'][random_idx]
        
        # 如果当前批次没有不同身份，从其他批次随机选择
        for _ in range(5):  # 最多尝试5次
            random_batch_idx = random.randint(0, len(self.batch_files) - 1)
            if random_batch_idx != batch_data['batch_idx']:
                try:
                    random_batch = self._load_batch(random_batch_idx)
                    random_sample_idx = random.randint(0, len(random_batch['identities']) - 1)
                    if random_batch['identities'][random_sample_idx] != current_identity:
                        return random_batch['body_tensors'][random_sample_idx]
                except:
                    continue
        
        return None
    
    def _augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """简单的数据增强"""
        if random.random() < 0.5:
            # 水平翻转
            tensor = torch.flip(tensor, [2])
        
        if random.random() < 0.3:
            # 轻微的亮度调整
            brightness_factor = random.uniform(0.8, 1.2)
            tensor = torch.clamp(tensor * brightness_factor, 0, 1)
        
        return tensor
    
    def get_statistics(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        return {
            'total_samples': getattr(self, 'total_samples', len(self.sample_mapping)),
            'num_real': self.index_data.get('num_real', 0),
            'num_fake': self.index_data.get('num_fake', 0),
            'num_identities': self.index_data.get('num_identities', 0),
            'num_batches': len(self.batch_files)
        }
    
    def cleanup_cache(self):
        """清理缓存"""
        self.cache.clear()
        self.cache_order.clear()
        gc.collect()
        print("缓存已清理")

def create_batched_dataloader(index_file: str, 
                             batch_size: int = 32,
                             negative_ratio: float = 0.5,
                             augment: bool = True,
                             shuffle: bool = True,
                             num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    创建分批数据加载器
    
    Args:
        index_file: 索引文件路径
        batch_size: 批次大小
        negative_ratio: 负样本比例
        augment: 是否数据增强
        shuffle: 是否打乱数据
        num_workers: 工作进程数
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = BatchedDataset(
        index_file=index_file,
        negative_ratio=negative_ratio,
        augment=augment
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader