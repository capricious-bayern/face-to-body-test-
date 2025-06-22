#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备Celeb-DF-v2数据集用于训练
处理所有真实和合成视频，提取人脸和身体特征
"""

import os
import torch
import numpy as np
from process_celeb_df import process_celeb_df_dataset
from dataset import FaceBodyDataset
from torch.utils.data import DataLoader
import pickle
from datetime import datetime
import gc

def prepare_data(dataset_path=None):
    """
    准备Celeb-DF-v2数据集
    
    Args:
        dataset_path: 数据集路径（可选）
    """
    print("开始处理Celeb-DF-v2数据集...")
    print("="*60)
    
    # 处理视频数据
    try:
        batch_files = process_celeb_df_dataset(dataset_path)  # 传递数据集路径
    except Exception as e:
        print(f"处理视频数据时出错: {e}")
        return None
    
    if len(batch_files) == 0:
        print("未提取到任何有效数据")
        return None
    
    print(f"\n获得 {len(batch_files)} 个批次文件，开始分批处理...")
    
    # 创建最终保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "processed_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # 分批处理并保存最终数据文件
    final_batch_files = []
    total_samples = 0
    total_real = 0
    total_fake = 0
    all_identities = set()
    
    for batch_idx, batch_file in enumerate(batch_files):
        print(f"\n处理批次 {batch_idx + 1}/{len(batch_files)}: {os.path.basename(batch_file)}")
        
        # 加载批次数据
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        faces = batch_data['faces']
        bodies = batch_data['bodies']
        labels = batch_data['labels']
        identities = batch_data['identities']
        
        print(f"批次样本数: {len(faces)}")
        
        # 转换为张量
        face_tensors = []
        body_tensors = []
        
        for face, body in zip(faces, bodies):
            # 转换为张量 (H, W, C) -> (C, H, W)
            face_tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
            body_tensor = torch.from_numpy(body).permute(2, 0, 1).float() / 255.0
            
            face_tensors.append(face_tensor)
            body_tensors.append(body_tensor)
        
        # 保存处理后的批次数据
        final_batch_file = os.path.join(save_dir, f"final_batch_{batch_idx}_{timestamp}.pkl")
        final_data = {
            'face_tensors': face_tensors,
            'body_tensors': body_tensors,
            'labels': labels,
            'identities': identities,
            'batch_idx': batch_idx,
            'num_samples': len(faces)
        }
        
        with open(final_batch_file, 'wb') as f:
            pickle.dump(final_data, f)
        
        final_batch_files.append(final_batch_file)
        
        # 统计信息
        total_samples += len(faces)
        total_real += sum(1 for l in labels if l == 0)
        total_fake += sum(1 for l in labels if l == 1)
        all_identities.update(identities)
        
        print(f"批次 {batch_idx + 1} 处理完成，保存到: {final_batch_file}")
        
        # 清理内存
        del faces, bodies, face_tensors, body_tensors, batch_data, final_data
        gc.collect()
        
        # 删除原始临时文件
        os.remove(batch_file)
        print(f"已删除临时文件: {batch_file}")
    
    # 创建索引文件
    index_file = os.path.join(save_dir, f"celeb_df_index_{timestamp}.pkl")
    index_data = {
        'batch_files': final_batch_files,
        'num_batches': len(final_batch_files),
        'total_samples': total_samples,
        'num_real': total_real,
        'num_fake': total_fake,
        'num_identities': len(all_identities),
        'timestamp': timestamp
    }
    
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"\n数据处理完成:")
    print(f"- 总样本数: {total_samples}")
    print(f"- 真实样本: {total_real}")
    print(f"- 合成样本: {total_fake}")
    print(f"- 身份数量: {len(all_identities)}")
    print(f"- 批次文件数: {len(final_batch_files)}")
    print(f"- 索引文件: {index_file}")
    
    return index_file

def load_processed_data(index_file):
    """
    加载处理后的数据索引
    
    Args:
        index_file: 索引文件路径
    
    Returns:
        dict: 包含批次文件信息的字典
    """
    print(f"加载数据索引文件: {index_file}")
    
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    
    print(f"数据索引加载完成:")
    print(f"- 总样本数: {index_data['total_samples']}")
    print(f"- 真实样本: {index_data['num_real']}")
    print(f"- 合成样本: {index_data['num_fake']}")
    print(f"- 身份数量: {index_data['num_identities']}")
    print(f"- 批次文件数: {index_data['num_batches']}")
    print(f"- 处理时间: {index_data['timestamp']}")
    
    return index_data

def load_batch_data(batch_file):
    """
    加载单个批次数据
    
    Args:
        batch_file: 批次文件路径
    
    Returns:
        dict: 批次数据
    """
    with open(batch_file, 'rb') as f:
        return pickle.load(f)

def main():
    """
    主函数
    """
    print("Celeb-DF-v2 数据集准备工具")
    print("="*60)
    
    # 检查数据集路径
    dataset_path = r"D:\Dataset\Celeb-DF-v2"
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请确保Celeb-DF-v2数据集已下载到指定路径")
        return
    
    real_path = os.path.join(dataset_path, "Celeb-real")
    synthesis_path = os.path.join(dataset_path, "Celeb-synthesis")
    
    if not os.path.exists(real_path):
        print(f"错误: Celeb-real文件夹不存在: {real_path}")
        return
    
    if not os.path.exists(synthesis_path):
        print(f"错误: Celeb-synthesis文件夹不存在: {synthesis_path}")
        return
    
    # 统计文件数量
    real_count = len([f for f in os.listdir(real_path) if f.lower().endswith('.mp4')])
    synthesis_count = len([f for f in os.listdir(synthesis_path) if f.lower().endswith('.mp4')])
    
    print(f"数据集统计:")
    print(f"- Celeb-real: {real_count} 个MP4文件")
    print(f"- Celeb-synthesis: {synthesis_count} 个MP4文件")
    print(f"- 总计: {real_count + synthesis_count} 个视频文件")
    
    if real_count == 0 and synthesis_count == 0:
        print("错误: 未找到任何MP4视频文件")
        return
    
    # 开始处理数据
    try:
        data_file = prepare_data()
        if data_file:
            print(f"\n数据准备完成！")
            print(f"处理后的数据文件: {data_file}")
            print(f"\n可以使用以下代码加载数据进行训练:")
            print(f"```python")
            print(f"from prepare_celeb_df_data import load_processed_data")
            print(f"data = load_processed_data('{data_file}')")
            print(f"```")
        else:
            print("数据处理失败")
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n程序结束")

if __name__ == "__main__":
    main()