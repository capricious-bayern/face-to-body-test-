#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频真假检测测试脚本

该脚本用于测试训练好的人脸-身体对应关系模型在原始MP4视频上的性能。
测试流程：
1. 从Celeb-DF-v2数据集中选择真实和合成视频各一个
2. 每隔30帧提取一帧进行处理
3. 使用YOLO和MTCNN检测并裁剪人脸和身体区域
4. 使用训练好的模型进行真假判断
5. 计算整体准确率
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import argparse
import yaml
from typing import Tuple, Optional, Dict, List
import time
from pathlib import Path
from tqdm import tqdm
import random

# 导入自定义模块
from models.face_encoder import create_face_encoder
from models.body_encoder import create_body_encoder
from models.classifier import create_classifier
from process_celeb_df import process_frame, extract_identity_from_filename
from facenet_pytorch import MTCNN
from ultralytics import YOLO

class VideoTester:
    """
    视频真假检测测试器
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'auto'):
        """
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            device: 设备类型 ('auto', 'cuda', 'cpu')
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 初始化检测模型
        self._init_detection_models()
        
        # 加载训练好的模型
        self._load_models(checkpoint_path)
        
        # 设置为评估模式
        self.face_encoder.eval()
        self.body_encoder.eval()
        self.classifier.eval()
        
        print("模型加载完成，准备开始测试")
    
    def _init_detection_models(self):
        """初始化检测模型"""
        print("正在加载检测模型...")
        self.mtcnn = MTCNN(keep_all=True, min_face_size=20, device=self.device)
        self.yolo_model = YOLO('yolov8n.pt')
        print("检测模型加载完成")
        
    def _load_models(self, checkpoint_path: str):
        """加载训练好的模型"""
        print(f"正在加载模型检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 准备人脸编码器参数
        face_encoder_params = self.config['model']['face_encoder']['params'].copy()
        if 'dropout' in face_encoder_params:
            face_encoder_params['dropout_rate'] = face_encoder_params.pop('dropout')
        
        # 准备身体编码器参数
        body_encoder_params = self.config['model']['body_encoder']['params'].copy()
        if 'dropout' in body_encoder_params:
            body_encoder_params['dropout_rate'] = body_encoder_params.pop('dropout')
        
        # 映射配置文件中的类型到create_body_encoder支持的类型
        body_type_mapping = {
            'BodyEncoder': 'basic',
            'MultiScaleBodyEncoder': 'multiscale', 
            'BodyPartEncoder': 'bodypart'
        }
        body_model_type = body_type_mapping.get(self.config['model']['body_encoder']['type'], 'basic')
        
        # 创建模型
        self.face_encoder = create_face_encoder(
            model_type=self.config['model']['face_encoder']['type'],
            **face_encoder_params
        ).to(self.device)
        
        self.body_encoder = create_body_encoder(
            model_type=body_model_type,
            **body_encoder_params
        ).to(self.device)
        
        self.classifier = create_classifier(
            classifier_type=self.config['model']['classifier']['type'],
            **self.config['model']['classifier']['params']
        ).to(self.device)
        
        # 加载权重
        self.face_encoder.load_state_dict(checkpoint['face_encoder_state_dict'])
        self.body_encoder.load_state_dict(checkpoint['body_encoder_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        print("模型权重加载完成")
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        预处理单帧图像
        
        Args:
            frame: 输入帧 [H, W, 3]
            
        Returns:
            face_tensor: 人脸张量 [1, 3, 256, 256] 或 None
            body_tensor: 身体张量 [1, 3, 256, 256] 或 None
        """
        # 使用现有的处理逻辑
        face_img, body_img = process_frame(frame, self.mtcnn, self.yolo_model)
        
        if face_img is None or body_img is None:
            return None, None
        
        # 转换为tensor
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float() / 255.0
        body_tensor = torch.from_numpy(body_img).permute(2, 0, 1).float() / 255.0
        
        # 添加batch维度
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        body_tensor = body_tensor.unsqueeze(0).to(self.device)
        
        return face_tensor, body_tensor
    
    def predict_single_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        对单帧进行预测
        
        Args:
            frame: 输入帧
            
        Returns:
            result: 预测结果字典
        """
        # 预处理
        face_tensor, body_tensor = self.preprocess_frame(frame)
        
        if face_tensor is None or body_tensor is None:
            return {
                'success': False,
                'prediction': None,
                'confidence': 0.0,
                'message': '无法检测到人脸或身体'
            }
        
        # 推理
        with torch.no_grad():
            # 提取特征
            face_features = self.face_encoder(face_tensor)
            body_features = self.body_encoder(body_tensor)
            
            # 分类
            logits = self.classifier(face_features, body_features)
            probabilities = torch.sigmoid(logits)  # 使用sigmoid因为是二分类
            
            # 预测结果 (0=真实, 1=合成)
            prediction = (probabilities > 0.5).float().item()
            confidence = probabilities.item() if prediction == 1 else (1 - probabilities.item())
            
            return {
                'success': True,
                'prediction': int(prediction),
                'confidence': confidence,
                'raw_probability': probabilities.item()
            }
    
    def test_video(self, video_path: str, true_label: int, frame_interval: int = 30) -> Dict[str, any]:
        """
        测试单个视频
        
        Args:
            video_path: 视频文件路径
            true_label: 真实标签 (0=真实, 1=合成)
            frame_interval: 帧间隔
            
        Returns:
            result: 测试结果字典
        """
        print(f"\n正在测试视频: {os.path.basename(video_path)}")
        print(f"真实标签: {'合成' if true_label == 1 else '真实'}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'success': False,
                'message': f'无法打开视频: {video_path}'
            }
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
        
        predictions = []
        confidences = []
        processed_frames = 0
        failed_frames = 0
        
        frame_count = 0
        pbar = tqdm(total=total_frames//frame_interval, desc="处理帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔frame_interval帧处理一次
            if frame_count % frame_interval == 0:
                result = self.predict_single_frame(frame)
                
                if result['success']:
                    predictions.append(result['prediction'])
                    confidences.append(result['confidence'])
                    processed_frames += 1
                else:
                    failed_frames += 1
                
                pbar.update(1)
            
            frame_count += 1
        
        cap.release()
        pbar.close()
        
        if processed_frames == 0:
            return {
                'success': False,
                'message': '没有成功处理任何帧'
            }
        
        # 计算统计信息
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # 使用多数投票决定最终预测
        final_prediction = int(np.round(np.mean(predictions)))
        avg_confidence = np.mean(confidences)
        
        # 计算准确率
        correct = (final_prediction == true_label)
        frame_accuracy = np.mean(predictions == true_label)
        
        print(f"处理结果:")
        print(f"  成功处理帧数: {processed_frames}")
        print(f"  失败帧数: {failed_frames}")
        print(f"  最终预测: {'合成' if final_prediction == 1 else '真实'}")
        print(f"  平均置信度: {avg_confidence:.3f}")
        print(f"  视频级准确率: {'正确' if correct else '错误'}")
        print(f"  帧级准确率: {frame_accuracy:.3f}")
        
        return {
            'success': True,
            'video_path': video_path,
            'true_label': true_label,
            'final_prediction': final_prediction,
            'avg_confidence': avg_confidence,
            'video_correct': correct,
            'frame_accuracy': frame_accuracy,
            'processed_frames': processed_frames,
            'failed_frames': failed_frames,
            'all_predictions': predictions.tolist(),
            'all_confidences': confidences.tolist()
        }
    
    def run_test_suite(self, dataset_path: str, num_videos_per_type: int = 1) -> Dict[str, any]:
        """
        运行完整的测试套件
        
        Args:
            dataset_path: Celeb-DF-v2数据集路径
            num_videos_per_type: 每种类型选择的视频数量
            
        Returns:
            results: 测试结果
        """
        print(f"\n开始测试套件")
        print(f"数据集路径: {dataset_path}")
        print(f"每种类型测试视频数: {num_videos_per_type}")
        print("="*60)
        
        real_path = os.path.join(dataset_path, "Celeb-real")
        synthesis_path = os.path.join(dataset_path, "Celeb-synthesis")
        
        # 获取视频文件列表
        real_videos = []
        if os.path.exists(real_path):
            for file in os.listdir(real_path):
                if file.lower().endswith('.mp4'):
                    real_videos.append(os.path.join(real_path, file))
        
        synthesis_videos = []
        if os.path.exists(synthesis_path):
            for file in os.listdir(synthesis_path):
                if file.lower().endswith('.mp4'):
                    synthesis_videos.append(os.path.join(synthesis_path, file))
        
        print(f"找到 {len(real_videos)} 个真实视频")
        print(f"找到 {len(synthesis_videos)} 个合成视频")
        
        if len(real_videos) == 0 and len(synthesis_videos) == 0:
            return {
                'success': False,
                'message': f'在 {dataset_path} 中未找到视频文件'
            }
        
        # 随机选择视频进行测试
        selected_real = random.sample(real_videos, min(num_videos_per_type, len(real_videos)))
        selected_synthesis = random.sample(synthesis_videos, min(num_videos_per_type, len(synthesis_videos)))
        
        print(f"\n选择的测试视频:")
        print(f"真实视频: {[os.path.basename(v) for v in selected_real]}")
        print(f"合成视频: {[os.path.basename(v) for v in selected_synthesis]}")
        
        # 测试所有选中的视频
        all_results = []
        
        # 测试真实视频
        for video_path in selected_real:
            result = self.test_video(video_path, true_label=0)
            if result['success']:
                all_results.append(result)
        
        # 测试合成视频
        for video_path in selected_synthesis:
            result = self.test_video(video_path, true_label=1)
            if result['success']:
                all_results.append(result)
        
        if len(all_results) == 0:
            return {
                'success': False,
                'message': '没有成功测试任何视频'
            }
        
        # 计算总体统计
        video_correct = sum(r['video_correct'] for r in all_results)
        total_videos = len(all_results)
        video_accuracy = video_correct / total_videos
        
        all_frame_accuracies = [r['frame_accuracy'] for r in all_results]
        avg_frame_accuracy = np.mean(all_frame_accuracies)
        
        total_processed_frames = sum(r['processed_frames'] for r in all_results)
        total_failed_frames = sum(r['failed_frames'] for r in all_results)
        
        # 打印总体结果
        print("\n" + "="*60)
        print("测试结果总结")
        print("="*60)
        print(f"测试视频总数: {total_videos}")
        print(f"视频级准确率: {video_accuracy:.3f} ({video_correct}/{total_videos})")
        print(f"平均帧级准确率: {avg_frame_accuracy:.3f}")
        print(f"总处理帧数: {total_processed_frames}")
        print(f"总失败帧数: {total_failed_frames}")
        print(f"帧处理成功率: {total_processed_frames/(total_processed_frames+total_failed_frames):.3f}")
        
        # 按类型统计
        real_results = [r for r in all_results if r['true_label'] == 0]
        synthesis_results = [r for r in all_results if r['true_label'] == 1]
        
        if real_results:
            real_acc = np.mean([r['video_correct'] for r in real_results])
            print(f"真实视频准确率: {real_acc:.3f} ({sum(r['video_correct'] for r in real_results)}/{len(real_results)})")
        
        if synthesis_results:
            synth_acc = np.mean([r['video_correct'] for r in synthesis_results])
            print(f"合成视频准确率: {synth_acc:.3f} ({sum(r['video_correct'] for r in synthesis_results)}/{len(synthesis_results)})")
        
        return {
            'success': True,
            'video_accuracy': video_accuracy,
            'avg_frame_accuracy': avg_frame_accuracy,
            'total_videos': total_videos,
            'video_correct': video_correct,
            'total_processed_frames': total_processed_frames,
            'total_failed_frames': total_failed_frames,
            'individual_results': all_results,
            'real_accuracy': np.mean([r['video_correct'] for r in real_results]) if real_results else 0,
            'synthesis_accuracy': np.mean([r['video_correct'] for r in synthesis_results]) if synthesis_results else 0
        }

def main():
    parser = argparse.ArgumentParser(description='视频真假检测测试')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='模型检查点路径')
    parser.add_argument('--dataset', type=str, default=r'D:\Dataset\Celeb-DF-v2', help='数据集路径')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='设备类型')
    parser.add_argument('--num_videos', type=int, default=1, help='每种类型测试的视频数量')
    parser.add_argument('--frame_interval', type=int, default=30, help='帧采样间隔')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型检查点不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.dataset):
        print(f"错误: 数据集路径不存在: {args.dataset}")
        return
    
    try:
        # 创建测试器
        tester = VideoTester(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        # 运行测试
        results = tester.run_test_suite(
            dataset_path=args.dataset,
            num_videos_per_type=args.num_videos
        )
        
        if results['success']:
            print("\n测试完成！")
        else:
            print(f"\n测试失败: {results['message']}")
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()