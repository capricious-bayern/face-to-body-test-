#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查点模型测试脚本
测试训练好的人脸-身体伪造检测模型在真实数据上的性能
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
import time
from collections import defaultdict

# 导入自定义模块
from models.face_encoder import create_face_encoder
from models.body_encoder import create_body_encoder
from models.classifier import create_classifier
from process_celeb_df import process_frame
from facenet_pytorch import MTCNN
from ultralytics import YOLO

class CheckpointTester:
    """
    检查点模型测试器
    """
    
    def __init__(self, config_path: str, device: str = 'auto'):
        """
        Args:
            config_path: 配置文件路径
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
        
    def _init_detection_models(self):
        """初始化检测模型"""
        print("初始化检测模型...")
        self.mtcnn = MTCNN(keep_all=True, min_face_size=20, device=self.device)
        self.yolo_model = YOLO('yolov8n.pt')
        print("检测模型初始化完成")
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点模型
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            tuple: (face_encoder, body_encoder, classifier)
        """
        print(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 创建模型
        face_encoder = create_face_encoder(
            model_type=self.config['model']['face_encoder']['type'],
            **self.config['model']['face_encoder']['params']
        ).to(self.device)
        
        body_encoder = create_body_encoder(
            model_type=self.config['model']['body_encoder']['type'],
            **self.config['model']['body_encoder']['params']
        ).to(self.device)
        
        classifier = create_classifier(
            classifier_type=self.config['model']['classifier']['type'],
            **self.config['model']['classifier']['params']
        ).to(self.device)
        
        # 加载权重
        face_encoder.load_state_dict(checkpoint['face_encoder_state_dict'])
        body_encoder.load_state_dict(checkpoint['body_encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # 设置为评估模式
        face_encoder.eval()
        body_encoder.eval()
        classifier.eval()
        
        print("模型加载完成")
        return face_encoder, body_encoder, classifier
    
    def collect_test_videos(self, dataset_path: str, num_real: int = 50, num_fake: int = 50) -> Tuple[List[str], List[str]]:
        """
        收集测试视频
        
        Args:
            dataset_path: 数据集路径
            num_real: 真实视频数量
            num_fake: 伪造视频数量
            
        Returns:
            tuple: (real_videos, fake_videos)
        """
        dataset_path = Path(dataset_path)
        
        # 查找真实视频
        real_dir = dataset_path / "Celeb-real"
        fake_dir = dataset_path / "Celeb-synthesis"
        
        real_videos = []
        fake_videos = []
        
        # 收集真实视频
        if real_dir.exists():
            all_real = list(real_dir.glob("*.mp4"))
            real_videos = random.sample(all_real, min(num_real, len(all_real)))
            print(f"找到 {len(all_real)} 个真实视频，选择 {len(real_videos)} 个")
        else:
            print(f"警告: 真实视频目录不存在: {real_dir}")
        
        # 收集伪造视频
        if fake_dir.exists():
            all_fake = list(fake_dir.glob("*.mp4"))
            fake_videos = random.sample(all_fake, min(num_fake, len(all_fake)))
            print(f"找到 {len(all_fake)} 个伪造视频，选择 {len(fake_videos)} 个")
        else:
            print(f"警告: 伪造视频目录不存在: {fake_dir}")
        
        return real_videos, fake_videos
    
    def process_video(self, video_path: str, models: Tuple, frame_interval: int = 30, max_frames: int = 10) -> List[Dict]:
        """
        处理单个视频
        
        Args:
            video_path: 视频路径
            models: (face_encoder, body_encoder, classifier)
            frame_interval: 帧间隔
            max_frames: 最大处理帧数
            
        Returns:
            List[Dict]: 每帧的预测结果
        """
        face_encoder, body_encoder, classifier = models
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []
        
        results = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样帧
            if frame_count % frame_interval == 0:
                try:
                    # 预处理帧
                    face_img, body_img = process_frame(frame, self.mtcnn, self.yolo_model)
                    
                    if face_img is not None and body_img is not None:
                        # 转换为张量
                        face_tensor = torch.from_numpy(face_img).unsqueeze(0).to(self.device)
                        body_tensor = torch.from_numpy(body_img).unsqueeze(0).to(self.device)
                        
                        # 预测
                        with torch.no_grad():
                            face_features = face_encoder(face_tensor)
                            body_features = body_encoder(body_tensor)
                            logits = classifier(face_features, body_features)
                            prob = torch.sigmoid(logits).item()
                        
                        results.append({
                            'frame_idx': frame_count,
                            'probability': prob,
                            'prediction': 1 if prob > 0.5 else 0
                        })
                        
                        processed_frames += 1
                    
                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {e}")
            
            frame_count += 1
        
        cap.release()
        return results
    
    def evaluate_checkpoint(self, checkpoint_path: str, dataset_path: str, 
                          num_real: int = 50, num_fake: int = 50) -> Dict:
        """
        评估单个检查点
        
        Args:
            checkpoint_path: 检查点路径
            dataset_path: 数据集路径
            num_real: 真实视频数量
            num_fake: 伪造视频数量
            
        Returns:
            Dict: 评估结果
        """
        print(f"\n{'='*60}")
        print(f"评估检查点: {Path(checkpoint_path).name}")
        print(f"{'='*60}")
        
        # 加载模型
        models = self.load_checkpoint(checkpoint_path)
        
        # 收集测试视频
        real_videos, fake_videos = self.collect_test_videos(dataset_path, num_real, num_fake)
        
        if not real_videos and not fake_videos:
            print("错误: 没有找到测试视频")
            return {}
        
        all_predictions = []
        all_labels = []
        video_results = []
        
        # 处理真实视频
        print(f"\n处理 {len(real_videos)} 个真实视频...")
        for i, video_path in enumerate(real_videos):
            print(f"处理真实视频 {i+1}/{len(real_videos)}: {video_path.name}")
            
            frame_results = self.process_video(video_path, models)
            if frame_results:
                # 视频级别预测（取平均）
                video_prob = np.mean([r['probability'] for r in frame_results])
                video_pred = 1 if video_prob > 0.5 else 0
                
                video_results.append({
                    'video_path': str(video_path),
                    'true_label': 1,  # 真实视频
                    'prediction': video_pred,
                    'probability': video_prob,
                    'frame_count': len(frame_results)
                })
                
                # 收集帧级别预测
                for result in frame_results:
                    all_predictions.append(result['probability'])
                    all_labels.append(1)  # 真实标签
        
        # 处理伪造视频
        print(f"\n处理 {len(fake_videos)} 个伪造视频...")
        for i, video_path in enumerate(fake_videos):
            print(f"处理伪造视频 {i+1}/{len(fake_videos)}: {video_path.name}")
            
            frame_results = self.process_video(video_path, models)
            if frame_results:
                # 视频级别预测（取平均）
                video_prob = np.mean([r['probability'] for r in frame_results])
                video_pred = 1 if video_prob > 0.5 else 0
                
                video_results.append({
                    'video_path': str(video_path),
                    'true_label': 0,  # 伪造视频
                    'prediction': video_pred,
                    'probability': video_prob,
                    'frame_count': len(frame_results)
                })
                
                # 收集帧级别预测
                for result in frame_results:
                    all_predictions.append(result['probability'])
                    all_labels.append(0)  # 伪造标签
        
        # 计算指标
        metrics = self._calculate_metrics(all_predictions, all_labels, video_results)
        
        return {
            'checkpoint_path': checkpoint_path,
            'metrics': metrics,
            'video_results': video_results,
            'total_videos': len(video_results),
            'total_frames': len(all_predictions)
        }
    
    def _calculate_metrics(self, predictions: List[float], labels: List[int], 
                          video_results: List[Dict]) -> Dict:
        """
        计算评估指标
        
        Args:
            predictions: 预测概率列表
            labels: 真实标签列表
            video_results: 视频级别结果
            
        Returns:
            Dict: 评估指标
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # 帧级别指标
        frame_preds = (predictions > 0.5).astype(int)
        frame_accuracy = (frame_preds == labels).mean()
        
        # 视频级别指标
        video_preds = np.array([r['prediction'] for r in video_results])
        video_labels = np.array([r['true_label'] for r in video_results])
        video_accuracy = (video_preds == video_labels).mean()
        
        # 分类别统计
        real_videos = [r for r in video_results if r['true_label'] == 1]
        fake_videos = [r for r in video_results if r['true_label'] == 0]
        
        real_correct = sum(1 for r in real_videos if r['prediction'] == 1)
        fake_correct = sum(1 for r in fake_videos if r['prediction'] == 0)
        
        real_accuracy = real_correct / len(real_videos) if real_videos else 0
        fake_accuracy = fake_correct / len(fake_videos) if fake_videos else 0
        
        # 计算混淆矩阵
        tp = sum(1 for r in video_results if r['true_label'] == 1 and r['prediction'] == 1)
        tn = sum(1 for r in video_results if r['true_label'] == 0 and r['prediction'] == 0)
        fp = sum(1 for r in video_results if r['true_label'] == 0 and r['prediction'] == 1)
        fn = sum(1 for r in video_results if r['true_label'] == 1 and r['prediction'] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'frame_level': {
                'accuracy': float(frame_accuracy),
                'total_frames': len(predictions)
            },
            'video_level': {
                'accuracy': float(video_accuracy),
                'real_accuracy': float(real_accuracy),
                'fake_accuracy': float(fake_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'total_videos': len(video_results)
            },
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        }
    
    def compare_checkpoints(self, checkpoint_paths: List[str], dataset_path: str, 
                          num_real: int = 50, num_fake: int = 50) -> Dict:
        """
        比较多个检查点
        
        Args:
            checkpoint_paths: 检查点路径列表
            dataset_path: 数据集路径
            num_real: 真实视频数量
            num_fake: 伪造视频数量
            
        Returns:
            Dict: 比较结果
        """
        results = {}
        
        for checkpoint_path in checkpoint_paths:
            if Path(checkpoint_path).exists():
                result = self.evaluate_checkpoint(checkpoint_path, dataset_path, num_real, num_fake)
                results[Path(checkpoint_path).name] = result
            else:
                print(f"警告: 检查点文件不存在: {checkpoint_path}")
        
        # 生成比较报告
        self._print_comparison_report(results)
        
        return results
    
    def _print_comparison_report(self, results: Dict):
        """
        打印比较报告
        
        Args:
            results: 比较结果
        """
        print(f"\n{'='*80}")
        print("检查点比较报告")
        print(f"{'='*80}")
        
        if not results:
            print("没有有效的评估结果")
            return
        
        # 表头
        print(f"{'检查点':<25} {'视频准确率':<12} {'真实准确率':<12} {'伪造准确率':<12} {'F1分数':<10} {'总视频数':<8}")
        print("-" * 80)
        
        # 结果行
        for name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']['video_level']
                print(f"{name:<25} {metrics['accuracy']:<12.3f} {metrics['real_accuracy']:<12.3f} "
                      f"{metrics['fake_accuracy']:<12.3f} {metrics['f1_score']:<10.3f} {metrics['total_videos']:<8}")
        
        # 详细指标
        print(f"\n{'='*80}")
        print("详细指标")
        print(f"{'='*80}")
        
        for name, result in results.items():
            if 'metrics' in result:
                print(f"\n{name}:")
                metrics = result['metrics']
                
                print(f"  视频级别:")
                print(f"    准确率: {metrics['video_level']['accuracy']:.3f}")
                print(f"    精确率: {metrics['video_level']['precision']:.3f}")
                print(f"    召回率: {metrics['video_level']['recall']:.3f}")
                print(f"    F1分数: {metrics['video_level']['f1_score']:.3f}")
                
                print(f"  混淆矩阵:")
                cm = metrics['confusion_matrix']
                print(f"    TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}")
                
                print(f"  帧级别:")
                print(f"    准确率: {metrics['frame_level']['accuracy']:.3f}")
                print(f"    总帧数: {metrics['frame_level']['total_frames']}")

def main():
    """
    主函数
    """
    # 配置路径
    config_path = "config.yaml"
    dataset_path = r"D:\Dataset\Celeb-DF-v2"
    
    # 检查点路径
    checkpoint_paths = [
        r"d:\study\DL\TRAE test\Face to Body\checkpoints\checkpoint_epoch_30.pth",
        r"d:\study\DL\TRAE test\Face to Body\checkpoints\best_model.pth"
    ]
    
    # 创建测试器
    tester = CheckpointTester(config_path)
    
    # 比较检查点
    results = tester.compare_checkpoints(
        checkpoint_paths=checkpoint_paths,
        dataset_path=dataset_path,
        num_real=50,
        num_fake=50
    )
    
    # 保存结果
    output_path = "checkpoint_comparison_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()