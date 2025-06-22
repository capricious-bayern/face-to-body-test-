#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统功能测试脚本
用于验证模型加载、数据处理和基本推理功能
"""

import os
import sys
import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
import argparse
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
try:
    from models.face_encoder import create_face_encoder
    from models.body_encoder import create_body_encoder
    from models.classifier import create_classifier
    from models.contrastive_loss import create_contrastive_loss
    from datasets.video_dataset import VideoFaceBodyDataset, ContrastiveVideoDataset
    from process_celeb_df import process_frame
    from facenet_pytorch import MTCNN
    from ultralytics import YOLO
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖包已正确安装")
    sys.exit(1)

class SystemTester:
    """
    系统测试器
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化测试器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 测试结果
        self.test_results = {}
        
        # 加载配置
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print("✓ 配置文件加载成功")
            self.test_results['config_loading'] = True
        except Exception as e:
            print(f"✗ 配置文件加载失败: {e}")
            self.test_results['config_loading'] = False
            raise
    
    def test_model_creation(self):
        """测试模型创建"""
        print("\n=== 测试模型创建 ===")
        
        try:
            # 测试人脸编码器
            face_encoder_params = self.config['model']['face_encoder']['params'].copy()
            # 确保移除image_size，因为它不是create_face_encoder的直接参数
            if 'image_size' in face_encoder_params:
                del face_encoder_params['image_size']
            face_encoder = create_face_encoder(
                model_type=self.config['model']['face_encoder']['type'],
                **face_encoder_params
            )
            print("✓ 人脸编码器创建成功")
            
            # 测试身体编码器
            body_encoder_params = self.config['model']['body_encoder']['params'].copy()
            # 确保移除image_size，因为它不是create_body_encoder的直接参数
            if 'image_size' in body_encoder_params:
                del body_encoder_params['image_size']
            body_encoder = create_body_encoder(
                model_type=self.config['model']['body_encoder']['type'],
                **body_encoder_params
            )
            print("✓ 身体编码器创建成功")
            
            # 测试分类器
            classifier = create_classifier(
                classifier_type=self.config['model']['classifier']['type'],
                **self.config['model']['classifier']['params']
            )
            print("✓ 分类器创建成功")
            
            self.test_results['model_creation'] = True
            return face_encoder, body_encoder, classifier
            
        except Exception as e:
            print(f"✗ 模型创建失败: {e}")
            self.test_results['model_creation'] = False
            return None, None, None
    
    def test_loss_functions(self):
        """测试损失函数"""
        print("\n=== 测试损失函数 ===")
        
        try:
            # 测试对比学习损失
            contrastive_loss = create_contrastive_loss(
                loss_type=self.config['loss']['contrastive']['type'],
                **self.config['loss']['contrastive']['params']
            )
            print("✓ 对比学习损失函数创建成功")
            
            # 测试分类损失
            classification_loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.config['loss']['classification']['params']['weight']),
                label_smoothing=self.config['loss']['classification']['params']['label_smoothing']
            )
            print("✓ 分类损失函数创建成功")
            
            self.test_results['loss_functions'] = True
            return contrastive_loss, classification_loss
            
        except Exception as e:
            print(f"✗ 损失函数创建失败: {e}")
            self.test_results['loss_functions'] = False
            return None, None
    
    def test_detection_models(self):
        """测试检测模型"""
        print("\n=== 测试检测模型 ===")
        
        try:
            # 测试MTCNN
            mtcnn = MTCNN(keep_all=True, min_face_size=20, device=self.device)
            print("✓ MTCNN模型加载成功")
            
            # 测试YOLO
            yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO模型加载成功")
            
            self.test_results['detection_models'] = True
            return mtcnn, yolo_model
            
        except Exception as e:
            print(f"✗ 检测模型加载失败: {e}")
            self.test_results['detection_models'] = False
            return None, None
    
    def test_data_processing(self, mtcnn, yolo_model):
        """测试数据处理"""
        print("\n=== 测试数据处理 ===")
        
        try:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 测试帧处理
            face_img, body_img = process_frame(test_image, mtcnn, yolo_model)
            
            if face_img is not None and body_img is not None:
                print(f"✓ 数据处理成功 - 人脸图像: {face_img.shape}, 身体图像: {body_img.shape}")
                self.test_results['data_processing'] = True
            else:
                print("⚠ 数据处理完成，但未检测到人脸或身体（正常情况，因为是随机图像）")
                self.test_results['data_processing'] = True
            
        except Exception as e:
            print(f"✗ 数据处理失败: {e}")
            self.test_results['data_processing'] = False
    
    def test_model_forward(self, face_encoder, body_encoder, classifier):
        """测试模型前向传播"""
        print("\n=== 测试模型前向传播 ===")
        
        try:
            # 创建测试数据
            batch_size = 2
            face_input = torch.randn(batch_size, 3, 256, 256).to(self.device)
            body_input = torch.randn(batch_size, 3, 256, 256).to(self.device)
            
            # 移动模型到设备
            face_encoder = face_encoder.to(self.device)
            body_encoder = body_encoder.to(self.device)
            classifier = classifier.to(self.device)
            
            # 设置为评估模式
            face_encoder.eval()
            body_encoder.eval()
            classifier.eval()
            
            with torch.no_grad():
                # 测试特征提取
                face_features = face_encoder.get_features(face_input)
                body_features = body_encoder.get_features(body_input)
                
                print(f"✓ 特征提取成功 - 人脸特征: {face_features.shape}, 身体特征: {body_features.shape}")
                
                # 测试分类
                logits = classifier(face_features, body_features)
                print(f"✓ 分类成功 - 输出形状: {logits.shape}")
                
                # 测试投影特征（用于对比学习）
                face_proj = face_encoder.get_projection(face_input)
                body_proj = body_encoder.get_projection(body_input)
                
                print(f"✓ 投影特征提取成功 - 人脸投影: {face_proj.shape}, 身体投影: {body_proj.shape}")
            
            self.test_results['model_forward'] = True
            
        except Exception as e:
            print(f"✗ 模型前向传播失败: {e}")
            self.test_results['model_forward'] = False
    
    def test_loss_computation(self, contrastive_loss, classification_loss):
        """测试损失计算"""
        print("\n=== 测试损失计算 ===")
        
        try:
            batch_size = 4
            feature_dim = self.config['model']['face_encoder']['params']['projection_dim']
            
            # 创建测试数据
            face_features = torch.randn(batch_size, feature_dim)
            body_features = torch.randn(batch_size, feature_dim)
            labels = torch.randint(0, 2, (batch_size,))
            identities = torch.randint(0, 10, (batch_size,))
            
            # 测试对比学习损失
            if hasattr(contrastive_loss, 'forward'):
                if 'IdentityAware' in self.config['loss']['contrastive']['type']:
                    cont_loss = contrastive_loss(face_features, body_features, identities, labels)
                else:
                    cont_loss = contrastive_loss(face_features, body_features, labels)
                print(f"✓ 对比学习损失计算成功: {cont_loss.item():.4f}")
            
            # 测试分类损失
            logits = torch.randn(batch_size, 2)
            class_loss = classification_loss(logits, labels)
            print(f"✓ 分类损失计算成功: {class_loss.item():.4f}")
            
            self.test_results['loss_computation'] = True
            
        except Exception as e:
            print(f"✗ 损失计算失败: {e}")
            self.test_results['loss_computation'] = False
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        print("\n=== 测试数据集创建 ===")
        
        try:
            # 检查数据集路径是否存在
            dataset_path = self.config['data']['dataset_path']
            if not os.path.exists(dataset_path):
                print(f"⚠ 数据集路径不存在: {dataset_path}")
                print("跳过数据集测试")
                self.test_results['dataset_creation'] = 'skipped'
                return
            
            # 创建测试数据集（只处理少量文件）
            dataset = VideoFaceBodyDataset(
                dataset_path=dataset_path,
                frame_interval=self.config['data']['frame_interval'],
                image_size=tuple(self.config['data']['image_size']),
                max_videos=2  # 只测试2个视频
            )
            
            print(f"✓ 数据集创建成功 - 包含 {len(dataset)} 个样本")
            
            # 测试数据加载
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✓ 数据加载成功 - 样本键: {list(sample.keys())}")
            
            self.test_results['dataset_creation'] = True
            
        except Exception as e:
            print(f"✗ 数据集创建失败: {e}")
            self.test_results['dataset_creation'] = False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始系统功能测试...")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 1. 测试模型创建
        face_encoder, body_encoder, classifier = self.test_model_creation()
        
        # 2. 测试损失函数
        contrastive_loss, classification_loss = self.test_loss_functions()
        
        # 3. 测试检测模型
        mtcnn, yolo_model = self.test_detection_models()
        
        # 4. 测试数据处理
        if mtcnn is not None and yolo_model is not None:
            self.test_data_processing(mtcnn, yolo_model)
        
        # 5. 测试模型前向传播
        if all([face_encoder, body_encoder, classifier]):
            self.test_model_forward(face_encoder, body_encoder, classifier)
        
        # 6. 测试损失计算
        if all([contrastive_loss, classification_loss]):
            self.test_loss_computation(contrastive_loss, classification_loss)
        
        # 7. 测试数据集创建
        self.test_dataset_creation()
        
        # 输出测试结果
        self.print_test_summary()
    
    def print_test_summary(self):
        """打印测试总结"""
        print("\n" + "="*50)
        print("测试结果总结")
        print("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result is True)
        skipped_tests = sum(1 for result in self.test_results.values() 
                           if result == 'skipped')
        failed_tests = total_tests - passed_tests - skipped_tests
        
        for test_name, result in self.test_results.items():
            status = "✓ 通过" if result is True else "⚠ 跳过" if result == 'skipped' else "✗ 失败"
            print(f"{test_name:20s}: {status}")
        
        print("-" * 50)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"跳过: {skipped_tests}")
        print(f"失败: {failed_tests}")
        
        if failed_tests == 0:
            print("\n🎉 所有测试通过！系统准备就绪。")
        else:
            print(f"\n⚠ {failed_tests} 个测试失败，请检查相关配置和依赖。")
        
        return failed_tests == 0

def main():
    parser = argparse.ArgumentParser(description='系统功能测试')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        sys.exit(1)
    
    # 运行测试
    tester = SystemTester(args.config)
    success = tester.run_all_tests()
    
    # 退出码
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()