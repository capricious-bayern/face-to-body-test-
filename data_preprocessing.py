#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
用于从视频中提取人脸和身体图像
"""

import cv2
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics import YOLO
from tqdm import tqdm
import torchvision.transforms as transforms

class VideoProcessor:
    """
    视频处理器，用于从视频中提取人脸和身体图像
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化视频处理器
        
        Args:
            device: 计算设备
        """
        self.device = device
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        print("正在加载检测模型...")
        self.mtcnn = MTCNN(keep_all=True, min_face_size=20, device=device)
        self.yolo_model = YOLO('yolov8n.pt')
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def process_video(self, video_path, frame_interval=30, max_frames=None):
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            frame_interval: 帧间隔
            max_frames: 最大处理帧数
            
        Returns:
            tuple: (face_tensors, body_tensors) 列表
        """
        print(f"处理视频: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return [], []
        
        face_tensors = []
        body_tensors = []
        frame_count = 0
        extracted_count = 0
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames = total_frames // frame_interval
        if max_frames:
            expected_frames = min(expected_frames, max_frames)
        
        with tqdm(total=expected_frames, desc="提取帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每隔指定帧数提取一帧
                if frame_count % frame_interval == 0:
                    face_tensor, body_tensor = self.process_frame(frame)
                    if face_tensor is not None and body_tensor is not None:
                        face_tensors.append(face_tensor)
                        body_tensors.append(body_tensor)
                        extracted_count += 1
                    
                    pbar.update(1)
                    
                    # 检查是否达到最大帧数
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        cap.release()
        print(f"成功提取了 {extracted_count} 对图像")
        
        return face_tensors, body_tensors
    
    def process_frame(self, frame):
        """
        处理单帧图像，裁剪人脸和身体区域
        
        Args:
            frame: 输入帧 (BGR格式)
            
        Returns:
            tuple: (face_tensor, body_tensor) 如果成功，否则 (None, None)
        """
        try:
            # 1. 使用YOLO检测person
            results = self.yolo_model(frame, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person类
                        confidence = box.conf[0]
                        if confidence >= 0.7:  # 置信度阈值
                            x1, y1, x2, y2 = [int(b) for b in box.xyxy[0]]
                            
                            # 裁剪人体区域
                            person_img = frame[y1:y2, x1:x2]
                            if person_img.size == 0:
                                continue
                            
                            # 2. 在人体区域上使用MTCNN检测人脸
                            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            person_pil = Image.fromarray(person_rgb)
                            boxes, _ = self.mtcnn.detect(person_pil)
                            
                            if boxes is not None and len(boxes) > 0:
                                # 取第一个检测到的人脸
                                face_box = boxes[0]
                                fx1, fy1, fx2, fy2 = [int(b) for b in face_box]
                                
                                # 确保人脸框在人体图像范围内
                                fx1 = max(0, fx1)
                                fy1 = max(0, fy1)
                                fx2 = min(person_img.shape[1], fx2)
                                fy2 = min(person_img.shape[0], fy2)
                                
                                # 3. 裁剪人脸区域
                                face_img = person_rgb[fy1:fy2, fx1:fx2]
                                if face_img.size == 0:
                                    continue
                                
                                # 4. 裁剪身体区域（去除头部）
                                body_start_y = fy2  # 从人脸框底部开始
                                body_img = person_rgb[body_start_y:, :]  # 从人脸底部到人体底部
                                
                                if body_img.size == 0 or body_img.shape[0] < 50:  # 确保身体区域足够大
                                    continue
                                
                                # 5. 调整图像大小并保持宽高比
                                face_tensor = self.resize_and_pad(face_img)
                                body_tensor = self.resize_and_pad(body_img)
                                
                                return face_tensor, body_tensor
            
            return None, None
        
        except Exception as e:
            return None, None
    
    def resize_and_pad(self, img, target_size=(256, 256)):
        """
        调整图像大小并填充，保持宽高比不失真
        
        Args:
            img: 输入图像 (RGB格式)
            target_size: 目标尺寸
            
        Returns:
            torch.Tensor: 处理后的图像张量 [3, 256, 256]
        """
        h, w, _ = img.shape
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(img, (new_w, new_h))
        
        # 创建填充图像（黑色填充）
        padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # 计算填充位置（居中）
        start_y = (target_size[1] - new_h) // 2
        start_x = (target_size[0] - new_w) // 2
        
        # 放置缩放后的图像
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        # 转换为tensor并归一化
        tensor = torch.from_numpy(padded).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return tensor
    
    def process_dataset(self, dataset_path, frame_interval=30, max_videos=None, max_frames_per_video=None):
        """
        处理整个数据集
        
        Args:
            dataset_path: 数据集路径
            frame_interval: 帧间隔
            max_videos: 最大处理视频数
            max_frames_per_video: 每个视频最大处理帧数
            
        Returns:
            tuple: (all_face_tensors, all_body_tensors)
        """
        print(f"开始处理数据集: {dataset_path}")
        
        # 获取视频文件列表
        video_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
                    if max_videos and len(video_files) >= max_videos:
                        break
            if max_videos and len(video_files) >= max_videos:
                break
        
        if not video_files:
            print(f"在路径 {dataset_path} 中未找到MP4视频文件")
            return [], []
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        all_face_tensors = []
        all_body_tensors = []
        
        for video_idx, video_path in enumerate(video_files):
            print(f"\n处理视频 {video_idx + 1}/{len(video_files)}")
            
            face_tensors, body_tensors = self.process_video(
                video_path, frame_interval, max_frames_per_video
            )
            
            all_face_tensors.extend(face_tensors)
            all_body_tensors.extend(body_tensors)
        
        print(f"\n数据集处理完成！总共提取了 {len(all_face_tensors)} 对图像")
        
        return all_face_tensors, all_body_tensors

def main():
    """
    测试数据预处理功能
    """
    dataset_path = r"D:\Dataset\Celeb-DF-v2"
    
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return
    
    processor = VideoProcessor()
    
    # 处理少量视频进行测试
    face_tensors, body_tensors = processor.process_dataset(
        dataset_path, 
        frame_interval=30, 
        max_videos=2, 
        max_frames_per_video=10
    )
    
    if face_tensors and body_tensors:
        print(f"\n测试结果:")
        print(f"人脸张量形状: {face_tensors[0].shape}")
        print(f"身体张量形状: {body_tensors[0].shape}")
        print(f"数据类型: {face_tensors[0].dtype}")
        print(f"数值范围: [{face_tensors[0].min():.3f}, {face_tensors[0].max():.3f}]")
    
if __name__ == "__main__":
    main()