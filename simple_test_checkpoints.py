#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版检查点测试脚本
使用现有的推理框架测试模型性能
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
import subprocess
import sys

def find_test_videos(dataset_path: str, num_real: int = 50, num_fake: int = 50) -> tuple:
    """
    查找测试视频
    
    Args:
        dataset_path: 数据集路径
        num_real: 真实视频数量
        num_fake: 伪造视频数量
        
    Returns:
        tuple: (real_videos, fake_videos)
    """
    dataset_path = Path(dataset_path)
    
    # 查找真实视频目录
    possible_real_dirs = [
        dataset_path / "Celeb-real",
        dataset_path / "real",
        dataset_path / "Real"
    ]
    
    # 查找伪造视频目录
    possible_fake_dirs = [
        dataset_path / "Celeb-synthesis", 
        dataset_path / "fake",
        dataset_path / "Fake",
        dataset_path / "synthesis"
    ]
    
    real_videos = []
    fake_videos = []
    
    # 查找真实视频
    for real_dir in possible_real_dirs:
        if real_dir.exists():
            print(f"找到真实视频目录: {real_dir}")
            all_real = list(real_dir.glob("*.mp4"))
            if all_real:
                selected_real = random.sample(all_real, min(num_real, len(all_real)))
                real_videos.extend(selected_real)
                print(f"从 {len(all_real)} 个真实视频中选择了 {len(selected_real)} 个")
                break
    
    # 查找伪造视频
    for fake_dir in possible_fake_dirs:
        if fake_dir.exists():
            print(f"找到伪造视频目录: {fake_dir}")
            all_fake = list(fake_dir.glob("*.mp4"))
            if all_fake:
                selected_fake = random.sample(all_fake, min(num_fake, len(all_fake)))
                fake_videos.extend(selected_fake)
                print(f"从 {len(all_fake)} 个伪造视频中选择了 {len(selected_fake)} 个")
                break
    
    return real_videos, fake_videos

def test_single_checkpoint(checkpoint_path: str, test_videos: List[Path], 
                          true_labels: List[int], output_dir: str) -> Dict:
    """
    测试单个检查点
    
    Args:
        checkpoint_path: 检查点路径
        test_videos: 测试视频列表
        true_labels: 真实标签列表
        output_dir: 输出目录
        
    Returns:
        Dict: 测试结果
    """
    checkpoint_name = Path(checkpoint_path).stem
    print(f"\n{'='*60}")
    print(f"测试检查点: {checkpoint_name}")
    print(f"{'='*60}")
    
    results = []
    predictions = []
    
    # 为每个视频运行推理
    for i, (video_path, true_label) in enumerate(zip(test_videos, true_labels)):
        print(f"处理视频 {i+1}/{len(test_videos)}: {video_path.name}")
        
        # 构建推理命令
        output_file = Path(output_dir) / f"{checkpoint_name}_{video_path.stem}_result.json"
        
        cmd = [
            sys.executable, "inference.py",
            "--config", "config.yaml",
            "--checkpoint", checkpoint_path,
            "--input", str(video_path),
            "--output", str(output_file),
            "--frame_interval", "30"
        ]
        
        try:
            # 运行推理命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and output_file.exists():
                # 读取结果
                with open(output_file, 'r', encoding='utf-8') as f:
                    inference_result = json.load(f)
                
                # 提取预测结果
                if 'predictions' in inference_result and inference_result['predictions']:
                    # 计算平均概率
                    probs = [pred.get('probability', 0.5) for pred in inference_result['predictions']]
                    avg_prob = sum(probs) / len(probs)
                    prediction = 1 if avg_prob > 0.5 else 0
                else:
                    avg_prob = 0.5
                    prediction = 0
                
                results.append({
                    'video_path': str(video_path),
                    'true_label': true_label,
                    'prediction': prediction,
                    'probability': avg_prob,
                    'frame_count': len(inference_result.get('predictions', []))
                })
                
                predictions.append(prediction)
                
                print(f"  预测: {prediction} (概率: {avg_prob:.3f}), 真实: {true_label}")
                
            else:
                print(f"  推理失败: {result.stderr}")
                # 添加失败结果
                results.append({
                    'video_path': str(video_path),
                    'true_label': true_label,
                    'prediction': 0,
                    'probability': 0.5,
                    'frame_count': 0,
                    'error': result.stderr
                })
                predictions.append(0)
                
        except subprocess.TimeoutExpired:
            print(f"  推理超时")
            results.append({
                'video_path': str(video_path),
                'true_label': true_label,
                'prediction': 0,
                'probability': 0.5,
                'frame_count': 0,
                'error': 'timeout'
            })
            predictions.append(0)
        
        except Exception as e:
            print(f"  推理出错: {e}")
            results.append({
                'video_path': str(video_path),
                'true_label': true_label,
                'prediction': 0,
                'probability': 0.5,
                'frame_count': 0,
                'error': str(e)
            })
            predictions.append(0)
    
    # 计算指标
    metrics = calculate_metrics(predictions, true_labels, results)
    
    return {
        'checkpoint_name': checkpoint_name,
        'checkpoint_path': checkpoint_path,
        'metrics': metrics,
        'results': results
    }

def calculate_metrics(predictions: List[int], true_labels: List[int], 
                     detailed_results: List[Dict]) -> Dict:
    """
    计算评估指标
    
    Args:
        predictions: 预测标签列表
        true_labels: 真实标签列表
        detailed_results: 详细结果列表
        
    Returns:
        Dict: 评估指标
    """
    if not predictions or not true_labels:
        return {}
    
    # 基本指标
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    # 分类别统计
    real_indices = [i for i, label in enumerate(true_labels) if label == 1]
    fake_indices = [i for i, label in enumerate(true_labels) if label == 0]
    
    real_correct = sum(1 for i in real_indices if predictions[i] == 1)
    fake_correct = sum(1 for i in fake_indices if predictions[i] == 0)
    
    real_accuracy = real_correct / len(real_indices) if real_indices else 0
    fake_accuracy = fake_correct / len(fake_indices) if fake_indices else 0
    
    # 混淆矩阵
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    
    # 精确率、召回率、F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        },
        'total_videos': total,
        'correct_predictions': correct
    }

def print_comparison_report(all_results: List[Dict]):
    """
    打印比较报告
    
    Args:
        all_results: 所有检查点的结果
    """
    print(f"\n{'='*80}")
    print("检查点比较报告")
    print(f"{'='*80}")
    
    if not all_results:
        print("没有有效的评估结果")
        return
    
    # 表头
    print(f"{'检查点':<25} {'总准确率':<10} {'真实准确率':<12} {'伪造准确率':<12} {'F1分数':<10} {'总视频数':<8}")
    print("-" * 80)
    
    # 结果行
    for result in all_results:
        if 'metrics' in result and result['metrics']:
            metrics = result['metrics']
            name = result['checkpoint_name']
            print(f"{name:<25} {metrics['accuracy']:<10.3f} {metrics['real_accuracy']:<12.3f} "
                  f"{metrics['fake_accuracy']:<12.3f} {metrics['f1_score']:<10.3f} {metrics['total_videos']:<8}")
    
    # 详细指标
    print(f"\n{'='*80}")
    print("详细指标")
    print(f"{'='*80}")
    
    for result in all_results:
        if 'metrics' in result and result['metrics']:
            print(f"\n{result['checkpoint_name']}:")
            metrics = result['metrics']
            
            print(f"  准确率: {metrics['accuracy']:.3f} ({metrics['correct_predictions']}/{metrics['total_videos']})")
            print(f"  精确率: {metrics['precision']:.3f}")
            print(f"  召回率: {metrics['recall']:.3f}")
            print(f"  F1分数: {metrics['f1_score']:.3f}")
            print(f"  真实视频准确率: {metrics['real_accuracy']:.3f}")
            print(f"  伪造视频准确率: {metrics['fake_accuracy']:.3f}")
            
            cm = metrics['confusion_matrix']
            print(f"  混淆矩阵: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")

def main():
    """
    主函数
    """
    print("检查点模型测试脚本")
    print("="*50)
    
    # 配置
    dataset_path = r"D:\Dataset\Celeb-DF-v2"
    checkpoint_paths = [
        r"d:\study\DL\TRAE test\Face to Body\checkpoints\checkpoint_epoch_30.pth",
        r"d:\study\DL\TRAE test\Face to Body\checkpoints\best_model.pth"
    ]
    
    num_real = 25  # 减少数量以加快测试
    num_fake = 25
    
    # 创建输出目录
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # 查找测试视频
    print(f"\n从数据集查找测试视频: {dataset_path}")
    real_videos, fake_videos = find_test_videos(dataset_path, num_real, num_fake)
    
    if not real_videos and not fake_videos:
        print("错误: 没有找到测试视频")
        print("请检查数据集路径是否正确")
        return
    
    # 准备测试数据
    test_videos = real_videos + fake_videos
    true_labels = [1] * len(real_videos) + [0] * len(fake_videos)
    
    print(f"\n测试数据准备完成:")
    print(f"  真实视频: {len(real_videos)} 个")
    print(f"  伪造视频: {len(fake_videos)} 个")
    print(f"  总计: {len(test_videos)} 个")
    
    # 测试每个检查点
    all_results = []
    
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).exists():
            try:
                result = test_single_checkpoint(
                    checkpoint_path, test_videos, true_labels, str(output_dir)
                )
                all_results.append(result)
            except Exception as e:
                print(f"测试检查点 {checkpoint_path} 时出错: {e}")
        else:
            print(f"警告: 检查点文件不存在: {checkpoint_path}")
    
    # 生成比较报告
    print_comparison_report(all_results)
    
    # 保存结果
    results_file = output_dir / "checkpoint_comparison.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n详细结果已保存到: {results_file}")
    
    # 推荐最佳模型
    if all_results:
        best_result = max(all_results, key=lambda x: x.get('metrics', {}).get('accuracy', 0))
        print(f"\n推荐使用: {best_result['checkpoint_name']} (准确率: {best_result['metrics']['accuracy']:.3f})")

if __name__ == "__main__":
    main()