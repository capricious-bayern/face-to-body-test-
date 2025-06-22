# -*- coding: utf-8 -*-
"""
BatchedStreamDataset - 内存高效的视频流式数据集
用于从MP4视频中批式采样图像并训练，避免内存溢出
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics import YOLO
import random
import re
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

class BatchedStreamDataset(Dataset):
    """
    批式流数据集 - 从视频中实时采样图像对进行训练
    
    特点：
    - 不缓存所有帧，仅处理一小批后立即用于训练
    - 内存高效，避免OOM问题
    - 支持正负样本生成
    - GPU加速处理
    """
    
    def __init__(self, 
                 dataset_path: str = r"D:\Dataset\Celeb-DF-v2",
                 batch_size: int = 32,
                 frame_interval: int = 30,
                 image_size: int = 256,
                 negative_ratio: float = 0.3,
                 device: str = 'cuda'):
        """
        初始化数据集
        
        Args:
            dataset_path: Celeb-DF-v2数据集路径
            batch_size: 每批返回的样本数量
            frame_interval: 视频帧采样间隔
            image_size: 输出图像尺寸
            negative_ratio: 负样本比例
            device: 计算设备
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.negative_ratio = negative_ratio
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.yolo_model = YOLO('yolov8n.pt') # 确保YOLO模型在这里初始化
        self.yolo_model.to(self.device)

        self.current_cap = None # 初始化 current_cap
        print(f"初始化模型，使用设备: {self.device}")
        self.mtcnn = MTCNN(keep_all=True, min_face_size=20, device=self.device)

        
        # 获取视频文件列表
        self.video_files = self._get_video_files()
        print(f"找到 {len(self.video_files)} 个视频文件")
        
        # 当前视频索引和帧位置
        self.current_video_idx = 0
        self.current_frame_pos = 0
        self.current_cap = None
        
        # 内存缓存池
        self.cache_pool = []
        self.cache_size = batch_size * 2  # 缓存池大小
        
    def _get_video_files(self) -> List[Dict[str, Any]]:
        """获取所有视频文件信息"""
        video_files = []
        
        # 真实视频
        real_path = os.path.join(self.dataset_path, "Celeb-real")
        if os.path.exists(real_path):
            for file in os.listdir(real_path):
                if file.lower().endswith('.mp4'):
                    video_files.append({
                        'path': os.path.join(real_path, file),
                        'name': file,
                        'label': 0,  # 真实
                        'identity': self._extract_identity(file)
                    })
        
        # 合成视频
        synthesis_path = os.path.join(self.dataset_path, "Celeb-synthesis")
        if os.path.exists(synthesis_path):
            for file in os.listdir(synthesis_path):
                if file.lower().endswith('.mp4'):
                    video_files.append({
                        'path': os.path.join(synthesis_path, file),
                        'name': file,
                        'label': 1,  # 合成
                        'identity': self._extract_identity(file)
                    })
        
        # 随机打乱
        random.shuffle(video_files)
        return video_files
    
    def _extract_identity(self, filename: str) -> str:
        """从文件名提取身份ID"""
        # 匹配 id数字 格式
        match = re.search(r'id(\d+)', filename)
        if match:
            return f"id{match.group(1)}"
        return "unknown"
    
    def _open_next_video(self) -> bool:
        """打开下一个视频文件"""
        if self.current_cap is not None:
            self.current_cap.release()
            self.current_cap = None
        
        if self.current_video_idx >= len(self.video_files):
            # 重新开始
            self.current_video_idx = 0
            random.shuffle(self.video_files)
        
        video_info = self.video_files[self.current_video_idx]
        self.current_cap = cv2.VideoCapture(video_info['path'])
        self.current_frame_pos = 0
        
        if not self.current_cap.isOpened():
            print(f"无法打开视频: {video_info['path']}")
            self.current_video_idx += 1
            return self._open_next_video()
        
        return True
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理单帧图像，提取人脸和身体区域"""
        try:
            # YOLO检测人体
            results = self.yolo_model(frame, verbose=False)
            
            # 找到面积最大的person
            max_area = 0
            best_person_box = None
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0:  # person类
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
                                best_person_box = (int(x1), int(y1), int(x2), int(y2))
            
            if best_person_box is None:
                return None, None
            
            x1, y1, x2, y2 = best_person_box
            person_img = frame[y1:y2, x1:x2]
            
            # MTCNN检测人脸
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb_frame)
            
            if boxes is None or len(boxes) == 0:
                return None, None
            
            # 选择第一个人脸框
            face_box = boxes[0]
            fx1, fy1, fx2, fy2 = [int(coord) for coord in face_box]
            
            # 确保人脸框在person框内
            if not (x1 <= fx1 < fx2 <= x2 and y1 <= fy1 < fy2 <= y2):
                # 如果人脸不在person框内，使用person框上部作为人脸
                person_height = y2 - y1
                face_height = int(person_height * 0.3)  # 假设头部占30%
                fx1, fy1 = x1, y1
                fx2, fy2 = x2, y1 + face_height
            
            # 提取人脸
            face_img = frame[fy1:fy2, fx1:fx2]
            
            # 提取身体（去除头部）
            body_y_start = fy2  # 从人脸底部开始
            if body_y_start >= y2:
                body_y_start = y1 + int((y2 - y1) * 0.3)  # 如果计算有误，使用30%位置
            
            body_img = frame[body_y_start:y2, x1:x2]
            
            # 检查图像有效性
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                return None, None
            if body_img.shape[0] < 10 or body_img.shape[1] < 10:
                return None, None
            
            # Resize和Pad到目标尺寸
            face_processed = self._resize_and_pad(face_img)
            body_processed = self._resize_and_pad(body_img)
            
            return face_processed, body_processed
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return None, None
    
    def _resize_and_pad(self, img: np.ndarray) -> np.ndarray:
        """将图像resize并pad到目标尺寸，保持宽高比"""
        h, w = img.shape[:2]
        target_size = self.image_size
        
        # 计算缩放比例
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h))
        
        # 创建目标尺寸的黑色背景
        result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # 计算居中位置
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        
        # 将resize后的图像放到中心
        result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return result
    
    def _fill_cache_pool(self):
        """填充缓存池"""
        # 确保有打开的视频
        if self.current_cap is None or not self.current_cap.isOpened():
            if not self._open_next_video():
                return
        
        # 获取视频总帧数用于进度条
        total_frames = int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = self.video_files[self.current_video_idx]['name']
        
        with tqdm(total=total_frames // self.frame_interval, initial=self.current_frame_pos // self.frame_interval, 
                  desc=f"处理视频 {video_name}", unit="frame", leave=False) as pbar:
            while len(self.cache_pool) < self.cache_size:
                ret, frame = self.current_cap.read()
                if not ret:
                    # 视频结束，切换到下一个
                    self.current_video_idx += 1
                    self.current_frame_pos = 0
                    if not self._open_next_video():
                        break
                    total_frames = int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_name = self.video_files[self.current_video_idx]['name']
                    pbar.total = total_frames // self.frame_interval
                    pbar.n = 0
                    pbar.set_description(f"处理视频 {video_name}")
                    continue
                
                # 每隔frame_interval帧提取一帧
                if self.current_frame_pos % self.frame_interval == 0:
                    face_img, body_img = self._process_frame(frame)
                    if face_img is not None and body_img is not None:
                        video_info = self.video_files[self.current_video_idx]
                        
                        # 转换为tensor
                        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float() / 255.0
                        body_tensor = torch.from_numpy(body_img).permute(2, 0, 1).float() / 255.0
                        
                        sample = {
                            'face': face_tensor,
                            'body': body_tensor,
                            'video_label': video_info['label'],
                            'identity': f"{video_info['identity']}_{self.current_frame_pos}"
                        }
                        
                        self.cache_pool.append(sample)
                    pbar.update(1)
                
                self.current_frame_pos += 1
    
    def _generate_batch(self) -> Dict[str, torch.Tensor]:
        """生成一个批次的数据"""
        # 确保缓存池有足够数据
        self._fill_cache_pool()
        
        if len(self.cache_pool) < self.batch_size:
            # 如果缓存池数据不足，用现有数据填充
            batch_samples = self.cache_pool[:]
            self.cache_pool = []
            
            # 重复样本以达到batch_size
            while len(batch_samples) < self.batch_size:
                if len(batch_samples) == 0:
                    break
                batch_samples.extend(batch_samples[:min(len(batch_samples), 
                                                       self.batch_size - len(batch_samples))])
        else:
            # 从缓存池中取出batch_size个样本
            batch_samples = self.cache_pool[:self.batch_size]
            self.cache_pool = self.cache_pool[self.batch_size:]
        
        if len(batch_samples) == 0:
            return None
        
        # 生成正负样本
        faces = []
        bodies = []
        labels = []
        identities = []
        
        num_negative = int(len(batch_samples) * self.negative_ratio)
        
        for i, sample in enumerate(batch_samples):
            faces.append(sample['face'])
            identities.append(sample['identity'])
            
            if i < num_negative:
                # 负样本：随机选择不同的body
                random_idx = random.randint(0, len(batch_samples) - 1)
                while random_idx == i and len(batch_samples) > 1:
                    random_idx = random.randint(0, len(batch_samples) - 1)
                bodies.append(batch_samples[random_idx]['body'])
                labels.append(0)  # 不匹配
            else:
                # 正样本：匹配的face和body
                bodies.append(sample['body'])
                labels.append(1)  # 匹配
        
        # 转换为batch tensor
        face_batch = torch.stack(faces)
        body_batch = torch.stack(bodies)
        label_batch = torch.tensor(labels, dtype=torch.long)
        
        return {
            'face': face_batch,
            'body': body_batch,
            'label': label_batch,
            'identity': identities
        }
    
    def __len__(self) -> int:
        """返回数据集长度（估算）"""
        # 估算总帧数
        total_frames = len(self.video_files) * 1000  # 假设每个视频1000帧
        return total_frames // (self.frame_interval * self.batch_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个批次的数据"""
        batch = self._generate_batch()
        if batch is None:
            # 如果没有数据，返回空batch
            return {
                'face': torch.zeros(1, 3, self.image_size, self.image_size),
                'body': torch.zeros(1, 3, self.image_size, self.image_size),
                'label': torch.tensor([1], dtype=torch.long),
                'identity': ['empty']
            }
        return batch
    
    def __del__(self):
        """析构函数，释放资源"""
        if self.current_cap is not None:
            self.current_cap.release()


def test_dataset():
    """测试数据集"""
    print("测试 BatchedStreamDataset...")
    
    # 创建数据集
    dataset = BatchedStreamDataset(
        dataset_path=r"D:\Dataset\Celeb-DF-v2",
        batch_size=8,
        frame_interval=30,
        image_size=256,
        negative_ratio=0.3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"数据集长度: {len(dataset)}")
    
    # 测试获取数据
    for i in range(3):
        print(f"\n获取第 {i+1} 批数据...")
        batch = dataset[i]
        
        print(f"Face batch shape: {batch['face'].shape}")
        print(f"Body batch shape: {batch['body'].shape}")
        print(f"Label batch shape: {batch['label'].shape}")
        print(f"Labels: {batch['label'].tolist()}")
        print(f"Identities: {batch['identity'][:3]}...")  # 只显示前3个
        
        # 检查数据范围
        print(f"Face data range: [{batch['face'].min():.3f}, {batch['face'].max():.3f}]")
        print(f"Body data range: [{batch['body'].min():.3f}, {batch['body'].max():.3f}]")


if __name__ == "__main__":
    test_dataset()