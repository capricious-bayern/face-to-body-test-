#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理测试脚本
测试训练好的人脸-身体伪造检测模型
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

# 导入模型模块
from face_encoder import create_face_encoder
from body_encoder import create_body_encoder
from face_body_classifier import create_classifier
from dataset import create_mock_dataset

def load_model(checkpoint_path, device='cpu'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备
        
    Returns:
        tuple: (face_encoder, body_encoder, classifier)
    """
    print(f"加载模型检查点: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载训练配置
    config_path = Path('./checkpoints/config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 从配置中获取模型参数
    face_encoder_type = config.get('face_encoder_type', 'basic')
    body_encoder_type = config.get('body_encoder_type', 'basic')
    classifier_type = config.get('classifier_type', 'basic')
    face_backbone = config.get('face_backbone', 'mobilenet_v2')
    body_backbone = config.get('body_backbone', 'mobilenet_v2') # 默认为mobilenet_v2，因为efficientnet_pytorch可能未安装
    pretrained = config.get('pretrained', True)

    # 创建模型（使用训练时的配置）
    face_encoder = create_face_encoder(encoder_type=face_encoder_type, pretrained=pretrained)
    body_encoder = create_body_encoder(encoder_type=body_encoder_type, backbone=body_backbone, pretrained=pretrained)
    classifier = create_classifier(classifier_type=classifier_type)
    
    # 加载权重
    face_encoder.load_state_dict(checkpoint['face_encoder_state_dict'])
    body_encoder.load_state_dict(checkpoint['body_encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # 移动到设备
    face_encoder = face_encoder.to(device)
    body_encoder = body_encoder.to(device)
    classifier = classifier.to(device)
    
    # 设置为评估模式
    face_encoder.eval()
    body_encoder.eval()
    classifier.eval()
    
    print("模型加载完成")
    return face_encoder, body_encoder, classifier

def predict_batch(face_encoder, body_encoder, classifier, face_images, body_images):
    """
    批量预测
    
    Args:
        face_encoder: 人脸编码器
        body_encoder: 身体编码器
        classifier: 分类器
        face_images: 人脸图像 [B, 3, 256, 256]
        body_images: 身体图像 [B, 3, 256, 256]
        
    Returns:
        tuple: (predictions, face_embeddings, body_embeddings)
    """
    with torch.no_grad():
        # 提取特征
        face_embeddings = face_encoder(face_images)
        body_embeddings = body_encoder(body_images)
        
        # 分类预测
        predictions = classifier(face_embeddings, body_embeddings)
        predictions = torch.sigmoid(predictions)  # 转换为概率
        
        return predictions, face_embeddings, body_embeddings

def test_model_inference():
    """
    测试模型推理功能
    """
    print("=== 模型推理测试 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查点路径
    checkpoint_path = Path('./checkpoints/best_model.pth')
    if not checkpoint_path.exists():
        checkpoint_path = Path('./checkpoints/final_model.pth')
    
    if not checkpoint_path.exists():
        print("错误: 未找到模型检查点文件")
        return
    
    try:
        # 加载模型
        face_encoder, body_encoder, classifier = load_model(checkpoint_path, device)
        
        # 创建测试数据
        print("创建测试数据...")
        test_dataset = create_mock_dataset(num_samples=20, num_identities=5)
        
        # 测试几个样本
        test_samples = []
        for i in range(min(5, len(test_dataset))):
            sample = test_dataset[i]
            test_samples.append(sample)
        
        # 批量处理
        face_images = torch.stack([sample['face'] for sample in test_samples]).to(device)
        body_images = torch.stack([sample['body'] for sample in test_samples]).to(device)
        labels = torch.tensor([sample['label'] for sample in test_samples])
        identities = torch.tensor([sample['identity'] for sample in test_samples])
        
        print(f"测试批次大小: {face_images.shape[0]}")
        print(f"人脸图像形状: {face_images.shape}")
        print(f"身体图像形状: {body_images.shape}")
        
        # 预测
        predictions, face_embeddings, body_embeddings = predict_batch(
            face_encoder, body_encoder, classifier, face_images, body_images
        )
        
        print(f"\n=== 预测结果 ===")
        print(f"预测形状: {predictions.shape}")
        print(f"人脸特征形状: {face_embeddings.shape}")
        print(f"身体特征形状: {body_embeddings.shape}")
        
        # 显示详细结果
        for i in range(len(test_samples)):
            pred_prob = predictions[i].item()
            true_label = labels[i].item()
            identity = identities[i].item()
            
            pred_label = 1 if pred_prob > 0.5 else 0
            is_correct = pred_label == true_label
            
            print(f"\n样本 {i+1}:")
            print(f"  身份ID: {identity}")
            print(f"  真实标签: {true_label} ({'真实' if true_label == 1 else '伪造'})")
            print(f"  预测概率: {pred_prob:.4f}")
            print(f"  预测标签: {pred_label} ({'真实' if pred_label == 1 else '伪造'})")
            print(f"  预测正确: {'✓' if is_correct else '✗'}")
        
        # 计算准确率
        pred_labels = (predictions > 0.5).float().squeeze()
        accuracy = (pred_labels == labels.float().to(device)).float().mean().item()
        print(f"\n批次准确率: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # 特征相似度分析
        print(f"\n=== 特征分析 ===")
        
        # 计算特征的L2范数（应该接近1，因为已归一化）
        face_norms = torch.norm(face_embeddings, dim=1)
        body_norms = torch.norm(body_embeddings, dim=1)
        print(f"人脸特征L2范数: {face_norms.mean().item():.4f} ± {face_norms.std().item():.4f}")
        print(f"身体特征L2范数: {body_norms.mean().item():.4f} ± {body_norms.std().item():.4f}")
        
        # 计算同一身份的人脸-身体特征相似度
        face_body_sim = F.cosine_similarity(face_embeddings, body_embeddings, dim=1)
        print(f"人脸-身体余弦相似度: {face_body_sim.mean().item():.4f} ± {face_body_sim.std().item():.4f}")
        
        print("\n=== 推理测试完成 ===")
        
    except Exception as e:
        print(f"推理测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_training_results():
    """
    分析训练结果
    """
    print("\n=== 训练结果分析 ===")
    
    # 读取训练统计
    stats_path = Path('./checkpoints/training_stats.json')
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        epochs = stats['epoch']
        losses = stats['avg_loss']
        con_losses = stats['avg_contrastive_loss']
        cls_losses = stats['avg_classification_loss']
        accuracies = stats['accuracy']
        
        print(f"训练轮数: {len(epochs)}")
        print(f"最终损失: {losses[-1]:.4f}")
        print(f"最终对比损失: {con_losses[-1]:.4f}")
        print(f"最终分类损失: {cls_losses[-1]:.4f}")
        print(f"最终准确率: {accuracies[-1]:.4f} ({accuracies[-1]*100:.1f}%)")
        
        # 损失趋势
        if len(losses) > 1:
            loss_trend = "下降" if losses[-1] < losses[0] else "上升"
            acc_trend = "上升" if accuracies[-1] > accuracies[0] else "下降"
            print(f"损失趋势: {loss_trend}")
            print(f"准确率趋势: {acc_trend}")
    else:
        print("未找到训练统计文件")
    
    # 读取配置
    config_path = Path('./checkpoints/config.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n训练配置:")
        print(f"  数据类型: {'模拟数据' if config['use_mock_data'] else '真实数据'}")
        print(f"  样本数量: {config['mock_samples']}")
        print(f"  身份数量: {config['mock_identities']}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  学习率: {config['lr']}")
        print(f"  训练轮数: {config['epochs']}")
        print(f"  编码器类型: Face={config['face_encoder_type']}, Body={config['body_encoder_type']}")
        print(f"  分类器类型: {config['classifier_type']}")

if __name__ == "__main__":
    # 分析训练结果
    analyze_training_results()
    
    # 测试模型推理
    test_model_inference()