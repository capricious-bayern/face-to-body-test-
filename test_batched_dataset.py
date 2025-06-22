# -*- coding: utf-8 -*-
"""
测试 BatchedStreamDataset 的训练脚本
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from batched_stream_dataset import BatchedStreamDataset
import time
from tqdm import tqdm
import argparse

class SimpleFaceBodyMatcher(nn.Module):
    """简单的人脸-身体匹配网络"""
    
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # 人脸编码器
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # 256->128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128->64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 64->32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # 32->16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim)
        )
        
        # 身体编码器
        self.body_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # 256->128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128->64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 64->32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # 32->16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 匹配/不匹配
        )
    
    def forward(self, face, body):
        face_features = self.face_encoder(face)
        body_features = self.body_encoder(body)
        
        # 拼接特征
        combined_features = torch.cat([face_features, body_features], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        
        return logits, face_features, body_features


def train_model():
    """训练模型"""
    print("开始训练 BatchedStreamDataset...")
    print("="*60)
    
    # 设备 - 强制使用GPU加速
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请确保您的PyTorch环境已正确配置CUDA。")
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n创建训练数据集...")
    dataset = BatchedStreamDataset(
        dataset_path=r"D:\Dataset\Celeb-DF-v2",
        batch_size=16,  # 每批16个样本
        frame_interval=30,
        image_size=256,
        negative_ratio=0.4,
        device=device  # 确保数据集使用GPU
    )
    
    # 创建数据加载器（batch_size=1，因为dataset已经返回批次）
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型
    print("\n创建模型...")
    model = SimpleFaceBodyMatcher(feature_dim=512).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练参数 - 增加到30轮
    num_epochs = 30
    log_interval = 5
    
    print(f"\n开始训练 {num_epochs} 个epoch...")
    
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            epoch_correct = 0
            epoch_samples = 0
            
            # 使用tqdm显示进度
            pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch_data in pbar:
                try:
                    # 获取数据（注意：batch_data是包含一个批次的字典）
                    batch = batch_data[0] if isinstance(batch_data, list) else batch_data
                    
                    faces = batch['face'].to(device)  # [B, 3, 256, 256]
                    bodies = batch['body'].to(device)  # [B, 3, 256, 256]
                    labels = batch['label'].to(device)  # [B]
                    
                    # 检查数据形状
                    if faces.dim() == 5:  # [1, B, 3, 256, 256]
                        faces = faces.squeeze(0)
                        bodies = bodies.squeeze(0)
                        labels = labels.squeeze(0)
                    
                    batch_size = faces.size(0)
                    if batch_size == 0:
                        continue
                    
                    # 前向传播
                    optimizer.zero_grad()
                    logits, face_features, body_features = model(faces, bodies)
                    
                    # 计算损失
                    loss = criterion(logits, labels)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    # 统计
                    _, predicted = torch.max(logits.data, 1)
                    correct = (predicted == labels).sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    total_samples += batch_size
                    
                    epoch_loss += loss.item()
                    epoch_correct += correct
                    epoch_samples += batch_size
                    
                    # 更新进度条
                    if batch_idx % log_interval == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0
                        pbar.set_postfix({
                            'Loss': f'{avg_loss:.4f}',
                            'Acc': f'{accuracy:.3f}',
                            'Batch': f'{batch_size}'
                        })
                    
                    # 限制训练批次数量（避免过度训练）
                    if batch_idx >= 50:  # 每轮训练50个批次
                        break
                        
                except Exception as e:
                    print(f"\n批次 {batch_idx} 处理出错: {e}")
                    continue
            
            # Epoch统计
            epoch_avg_loss = epoch_loss / max(batch_idx + 1, 1)
            epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0
            
            print(f"\nEpoch {epoch + 1} 完成:")
            print(f"  平均损失: {epoch_avg_loss:.4f}")
            print(f"  准确率: {epoch_accuracy:.3f}")
            print(f"  处理样本数: {epoch_samples}")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 总体统计
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n训练完成!")
    print(f"总训练时间: {total_time:.2f} 秒")
    if total_samples > 0:
        print(f"总体准确率: {total_correct / total_samples:.3f}")
        print(f"平均损失: {total_loss / max(total_samples // 16, 1):.4f}")
        print(f"处理样本总数: {total_samples}")
    
    # 保存模型
    try:
        torch.save(model.state_dict(), 'batched_stream_model.pth')
        print(f"模型已保存到: batched_stream_model.pth")
    except Exception as e:
        print(f"保存模型时出错: {e}")


def create_test_dataset(dataset_path, num_files_per_class=50):
    """创建专门的测试数据集，从真实和合成视频各抽取指定数量的文件"""
    import os
    import random
    
    # 获取真实视频文件
    real_path = os.path.join(dataset_path, "Celeb-real")
    real_files = []
    if os.path.exists(real_path):
        all_real_files = [f for f in os.listdir(real_path) if f.lower().endswith('.mp4')]
        real_files = random.sample(all_real_files, min(num_files_per_class, len(all_real_files)))
    
    # 获取合成视频文件
    synthesis_path = os.path.join(dataset_path, "Celeb-synthesis")
    synthesis_files = []
    if os.path.exists(synthesis_path):
        all_synthesis_files = [f for f in os.listdir(synthesis_path) if f.lower().endswith('.mp4')]
        synthesis_files = random.sample(all_synthesis_files, min(num_files_per_class, len(all_synthesis_files)))
    
    print(f"测试集包含: {len(real_files)} 个真实视频, {len(synthesis_files)} 个合成视频")
    
    return real_files, synthesis_files

def evaluate_model(model_path="batched_stream_model.pth"):
    """评估模型性能 - 使用专门的测试集"""
    import os
    print("\n开始评估模型...")
    print("="*60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请确保您的PyTorch环境已正确配置CUDA。")
    device = torch.device('cuda')
    print(f"使用设备: {device}")

    # 创建专门的测试数据集
    print("\n创建测试数据集（各50个文件）...")
    real_files, synthesis_files = create_test_dataset(r"D:\Dataset\Celeb-DF-v2", 50)
    
    # 创建评估数据集
    dataset = BatchedStreamDataset(
        dataset_path=r"D:\Dataset\Celeb-DF-v2",
        batch_size=16,
        frame_interval=30,
        image_size=256,
        negative_ratio=0.5,
        device=device
    )
    
    # 手动设置测试文件列表
    test_video_files = []
    for file in real_files:
        test_video_files.append({
            'path': os.path.join(r"D:\Dataset\Celeb-DF-v2\Celeb-real", file),
            'name': file,
            'label': 0,  # 真实
            'identity': dataset._extract_identity(file)
        })
    for file in synthesis_files:
        test_video_files.append({
            'path': os.path.join(r"D:\Dataset\Celeb-DF-v2\Celeb-synthesis", file),
            'name': file,
            'label': 1,  # 合成
            'identity': dataset._extract_identity(file)
        })
    
    # 替换数据集的视频文件列表
    dataset.video_files = test_video_files
    dataset.current_video_idx = 0
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 加载模型
    print(f"\n加载模型: {model_path}...")
    model = SimpleFaceBodyMatcher(feature_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_correct = 0
    total_samples = 0
    real_correct = 0
    real_total = 0
    synthesis_correct = 0
    synthesis_total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), desc="评估进度")
        for batch_idx, batch_data in pbar:
            try:
                batch = batch_data[0] if isinstance(batch_data, list) else batch_data

                faces = batch['face'].to(device)
                bodies = batch['body'].to(device)
                labels = batch['label'].to(device)

                if faces.dim() == 5:
                    faces = faces.squeeze(0)
                    bodies = bodies.squeeze(0)
                    labels = labels.squeeze(0)

                batch_size = faces.size(0)
                if batch_size == 0:
                    continue

                logits, _, _ = model(faces, bodies)
                _, predicted = torch.max(logits.data, 1)
                
                correct = (predicted == labels).sum().item()
                total_correct += correct
                total_samples += batch_size
                
                # 分类统计
                for i in range(batch_size):
                    if labels[i] == 0:  # 真实
                        real_total += 1
                        if predicted[i] == labels[i]:
                            real_correct += 1
                    else:  # 合成
                        synthesis_total += 1
                        if predicted[i] == labels[i]:
                            synthesis_correct += 1

                pbar.set_postfix({
                    'Acc': f'{total_correct / total_samples:.3f}' if total_samples > 0 else 'N/A'
                })

                if batch_idx >= 30:  # 评估30个批次
                    break

            except Exception as e:
                print(f"\n评估批次 {batch_idx} 处理出错: {e}")
                continue

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    real_accuracy = real_correct / real_total if real_total > 0 else 0
    synthesis_accuracy = synthesis_correct / synthesis_total if synthesis_total > 0 else 0
    
    print(f"\n模型评估完成！")
    print(f"  总样本数: {total_samples}")
    print(f"  总正确数: {total_correct}")
    print(f"  总体准确率: {accuracy:.3f}")
    print(f"  真实视频准确率: {real_accuracy:.3f} ({real_correct}/{real_total})")
    print(f"  合成视频准确率: {synthesis_accuracy:.3f} ({synthesis_correct}/{synthesis_total})")

def test_dataset_only():
    """仅测试数据集功能"""
    print("\n仅测试数据集功能...")
    print("="*60)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请确保您的PyTorch环境已正确配置CUDA。")
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    dataset = BatchedStreamDataset(
        dataset_path=r"D:\Dataset\Celeb-DF-v2",
        batch_size=8, # 测试时可以小一点
        frame_interval=30,
        image_size=256,
        negative_ratio=0.5,
        device=device
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, batch_data in enumerate(dataloader):
        if i >= 3: # 只测试3个批次
            break
        
        batch = batch_data[0] if isinstance(batch_data, list) else batch_data
        
        faces = batch['face']
        bodies = batch['body']
        labels = batch['label']
        identities = batch['identity']
        
        print(f"\n获取第 {i+1} 批数据...")
        print(f"  Face batch shape: {faces.shape}")
        print(f"  Body batch shape: {bodies.shape}")
        print(f"  Label batch: {labels.tolist()}")
        print(f"  正样本数: {torch.sum(labels == 0).item()}")
        print(f"  负样本数: {torch.sum(labels == 1).item()}")
        print(f"  数据范围: Face[{faces.min():.3f}, {faces.max():.3f}], Body[{bodies.min():.3f}, {bodies.max():.3f}]")
        print(f"  身份示例: {identities.tolist()}")
        
    print("\n数据集测试完成！")


def predict_single_video(model_path="batched_stream_model.pth", video_path=None):
    """使用训练好的模型对单个视频进行预测"""
    print("\n开始单视频预测...")
    print("="*60)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，请确保您的PyTorch环境已正确配置CUDA。")
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {model_path}...")
    model = SimpleFaceBodyMatcher(feature_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 如果没有指定视频，随机选择一个
    if video_path is None:
        import os
        import random
        real_path = r"D:\Dataset\Celeb-DF-v2\Celeb-real"
        synthesis_path = r"D:\Dataset\Celeb-DF-v2\Celeb-synthesis"
        
        all_videos = []
        if os.path.exists(real_path):
            for f in os.listdir(real_path)[:5]:  # 只取前5个
                if f.lower().endswith('.mp4'):
                    all_videos.append((os.path.join(real_path, f), "真实", 0))
        if os.path.exists(synthesis_path):
            for f in os.listdir(synthesis_path)[:5]:  # 只取前5个
                if f.lower().endswith('.mp4'):
                    all_videos.append((os.path.join(synthesis_path, f), "合成", 1))
        
        if not all_videos:
            print("没有找到可用的视频文件！")
            return
        
        video_path, true_label_name, true_label = random.choice(all_videos)
        print(f"随机选择视频: {os.path.basename(video_path)} (真实标签: {true_label_name})")
    
    # 创建临时数据集来处理这个视频
    dataset = BatchedStreamDataset(
        dataset_path=r"D:\Dataset\Celeb-DF-v2",
        batch_size=8,
        frame_interval=30,
        image_size=256,
        negative_ratio=0.5,
        device=device
    )
    
    # 手动处理视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    predictions = []
    confidences = []
    frame_count = 0
    
    print("\n开始处理视频帧...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 != 0:  # 每30帧处理一次
                continue
            
            # 处理帧
            face_img, body_img = dataset._process_frame(frame)
            if face_img is None or body_img is None:
                continue
            
            # 预处理
            face_tensor = dataset._preprocess_image(face_img).unsqueeze(0).to(device)
            body_tensor = dataset._preprocess_image(body_img).unsqueeze(0).to(device)
            
            # 预测
            logits, _, _ = model(face_tensor, body_tensor)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            predictions.append(predicted.item())
            confidences.append(probs.max().item())
            
            if len(predictions) >= 10:  # 处理10帧就够了
                break
    
    cap.release()
    
    if not predictions:
        print("无法从视频中提取有效的人脸和身体区域！")
        return
    
    # 统计结果
    real_count = predictions.count(0)
    fake_count = predictions.count(1)
    avg_confidence = sum(confidences) / len(confidences)
    
    final_prediction = 0 if real_count > fake_count else 1
    final_label = "真实" if final_prediction == 0 else "合成"
    
    print(f"\n预测结果:")
    print(f"  视频文件: {os.path.basename(video_path)}")
    print(f"  处理帧数: {len(predictions)}")
    print(f"  预测为真实的帧数: {real_count}")
    print(f"  预测为合成的帧数: {fake_count}")
    print(f"  最终预测: {final_label}")
    print(f"  平均置信度: {avg_confidence:.3f}")
    if 'true_label_name' in locals():
        print(f"  真实标签: {true_label_name}")
        print(f"  预测正确: {'是' if final_prediction == true_label else '否'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练或测试人脸-身体匹配模型')
    parser.add_argument('--test-only', action='store_true', help='只测试数据集功能')
    parser.add_argument('--evaluate', action='store_true', help='评估模型性能')
    parser.add_argument('--predict', action='store_true', help='使用模型进行预测')
    parser.add_argument('--video-path', type=str, help='指定要预测的视频路径')
    args = parser.parse_args()
    
    if args.test_only:
        test_dataset_only()
    elif args.evaluate:
        evaluate_model()
    elif args.predict:
        predict_single_video(video_path=args.video_path)
    else:
        train_model()