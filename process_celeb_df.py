import cv2
import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics import YOLO
from tqdm import tqdm
import re
import pickle

def process_celeb_df_dataset(dataset_path=None):
    """
    处理Celeb-DF-v2数据集的视频文件
    
    Args:
        dataset_path: 数据集路径，如果为None则使用默认路径
    
    处理流程：
    1. 分别处理Celeb-real（真实）和Celeb-synthesis（合成）文件夹
    2. 每隔30帧提取一帧图像
    3. 使用YOLO检测person类，裁剪人体区域
    4. 使用MTCNN检测人脸区域
    5. 裁剪人脸和身体区域（去除头部）
    6. 根据文件名提取身份ID，正确标记真实/合成标签
    """
    
    # 设置路径
    if dataset_path is None:
        dataset_path = r"D:\Dataset\Celeb-DF-v2"
    real_path = os.path.join(dataset_path, "Celeb-real")
    synthesis_path = os.path.join(dataset_path, "Celeb-synthesis")
    
    # 检测设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    print("正在加载模型...")
    mtcnn = MTCNN(keep_all=True, min_face_size=20, device=device)
    yolo_model = YOLO('yolov8n.pt')
    
    # 获取真实视频文件列表
    real_videos = []
    if os.path.exists(real_path):
        for file in os.listdir(real_path):
            if file.lower().endswith('.mp4'):
                real_videos.append(os.path.join(real_path, file))
    
    # 获取合成视频文件列表 - 每个人物只选取前3个MP4文件
    synthesis_videos = []
    if os.path.exists(synthesis_path):
        # 按人物ID分组视频文件
        identity_videos = {}
        for file in os.listdir(synthesis_path):
            if file.lower().endswith('.mp4'):
                identity_id = extract_identity_from_filename(file)
                if identity_id not in identity_videos:
                    identity_videos[identity_id] = []
                identity_videos[identity_id].append(os.path.join(synthesis_path, file))
        
        # 每个人物只选取前3个视频文件
        for identity_id, videos in identity_videos.items():
            # 按文件名排序确保一致性
            videos.sort()
            # 只取前3个文件
            selected_videos = videos[:3]
            synthesis_videos.extend(selected_videos)
            print(f"人物ID {identity_id}: 找到 {len(videos)} 个视频，选择前 {len(selected_videos)} 个")
    
    print(f"找到 {len(real_videos)} 个真实视频文件")
    print(f"找到 {len(synthesis_videos)} 个合成视频文件")
    
    if not real_videos and not synthesis_videos:
        print(f"在路径 {dataset_path} 中未找到MP4视频文件")
        return
    
    all_faces = []
    all_bodies = []
    all_labels = []  # 0表示真实，1表示合成
    all_identities = []  # 身份ID
    

    processed_batches = []

    # 处理真实视频
    print("\n开始处理真实视频...")
    for video_idx, video_path in enumerate(real_videos):
        video_name = os.path.basename(video_path)
        print(f"\n处理真实视频 {video_idx + 1}/{len(real_videos)}: {video_name}")
        
        # 从文件名提取身份ID (格式: id1_id0_0000.mp4)
        identity_id = extract_identity_from_filename(video_name)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue
        
        frame_count = 0
        extracted_count = 0
        
        # 获取视频总帧数用于进度条
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames//30, desc=f"真实视频{video_idx+1}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每隔30帧提取一帧
                if frame_count % 30 == 0:
                    face_img, body_img = process_frame(frame, mtcnn, yolo_model)
                    if face_img is not None and body_img is not None:
                        all_faces.append(face_img)
                        all_bodies.append(body_img)
                        all_labels.append(0)  # 0表示真实
                        all_identities.append(identity_id)
                        extracted_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        print(f"从真实视频 {video_name} 中成功提取了 {extracted_count} 对图像，身份ID: {identity_id}")
        
        # 每个视频处理完成后保存数据并清空内存
        batch_data = {
            'faces': all_faces,
            'bodies': all_bodies,
            'labels': all_labels,
            'identities': all_identities
        }
        temp_file_path = f"processed_data/temp_batch_{len(processed_batches)}.pkl"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, 'wb') as f:
            pickle.dump(batch_data, f)
        processed_batches.append(temp_file_path)
        print(f"已保存临时批次文件: {temp_file_path}")
        
        # 清空内存
        all_faces = []
        all_bodies = []
        all_labels = []
        all_identities = []
    
    # 处理合成视频
    print("\n开始处理合成视频...")
    for video_idx, video_path in enumerate(synthesis_videos):
        video_name = os.path.basename(video_path)
        print(f"\n处理合成视频 {video_idx + 1}/{len(synthesis_videos)}: {video_name}")
        
        # 从文件名提取身份ID (格式: id1_id0_0000.mp4，取第一个ID作为主要身份)
        identity_id = extract_identity_from_filename(video_name)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue
        
        frame_count = 0
        extracted_count = 0
        
        # 获取视频总帧数用于进度条
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames//30, desc=f"合成视频{video_idx+1}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每隔30帧提取一帧
                if frame_count % 30 == 0:
                    face_img, body_img = process_frame(frame, mtcnn, yolo_model)
                    if face_img is not None and body_img is not None:
                        all_faces.append(face_img)
                        all_bodies.append(body_img)
                        all_labels.append(1)  # 1表示合成
                        all_identities.append(identity_id)
                        extracted_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        print(f"从合成视频 {video_name} 中成功提取了 {extracted_count} 对图像，身份ID: {identity_id}")
        
        # 每个视频处理完成后保存数据并清空内存
        batch_data = {
            'faces': all_faces,
            'bodies': all_bodies,
            'labels': all_labels,
            'identities': all_identities
        }
        temp_file_path = f"processed_data/temp_batch_{len(processed_batches)}.pkl"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, 'wb') as f:
            pickle.dump(batch_data, f)
        processed_batches.append(temp_file_path)
        print(f"已保存临时批次文件: {temp_file_path}")
        
        # 清空内存
        all_faces = []
        all_bodies = []
        all_labels = []
        all_identities = []
    
    print("\n数据预处理完成！")
    print(f"总共生成了 {len(processed_batches)} 个批次文件")
    
    # 统计总数据量（不加载到内存）
    total_samples = 0
    real_samples = 0
    fake_samples = 0
    unique_identities = set()
    
    print("\n正在统计数据...")
    for temp_file in processed_batches:
        with open(temp_file, 'rb') as f:
            batch_data = pickle.load(f)
            batch_size = len(batch_data['faces'])
            total_samples += batch_size
            real_samples += sum(1 for label in batch_data['labels'] if label == 0)
            fake_samples += sum(1 for label in batch_data['labels'] if label == 1)
            unique_identities.update(batch_data['identities'])
    
    print(f"总共提取了 {total_samples} 对图像")
    print(f"真实样本: {real_samples} 个")
    print(f"合成样本: {fake_samples} 个")
    print(f"涉及身份数量: {len(unique_identities)} 个")
    print(f"批次文件保存在: processed_data/ 目录")
    print("数据已分批保存，可以避免内存不足问题。")
    
    return processed_batches  # 返回批次文件路径列表而不是数据本身

def extract_identity_from_filename(filename):
    """
    从文件名中提取身份ID
    
    Args:
        filename: 文件名，格式如 "id1_id0_0000.mp4"
    
    Returns:
        int: 身份ID (取第一个id后的数字)
    """
    try:
        # 使用正则表达式提取第一个id后的数字
        match = re.search(r'id(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"警告: 无法从文件名 {filename} 中提取身份ID，使用默认值0")
            return 0
    except Exception as e:
        print(f"提取身份ID时出错: {e}，使用默认值0")
        return 0

def process_frame(frame, mtcnn, yolo_model):
    """
    处理单帧图像，裁剪人脸和身体区域并返回。
    
    Args:
        frame: 输入帧
        mtcnn: MTCNN模型
        yolo_model: YOLO模型
    
    Returns:
        tuple: (face_image, body_image) 如果成功裁剪，否则 (None, None)
    """
    try:
        # 1. 使用YOLO检测person
        results = yolo_model(frame, verbose=False)
        
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
                        person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                        boxes, _ = mtcnn.detect(person_pil)
                        
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
                            face_img = person_img[fy1:fy2, fx1:fx2]
                            if face_img.size == 0:
                                continue
                            
                            # 4. 裁剪身体区域（去除头部）
                            # 使用人脸框的底部作为身体的起始点
                            body_start_y = fy2  # 从人脸框底部开始
                            body_img = person_img[body_start_y:, :]  # 从人脸底部到人体底部
                            
                            if body_img.size == 0 or body_img.shape[0] < 50:  # 确保身体区域足够大
                                continue
                            
                            # 5. 调整图像大小并保存
                            # face_resized = cv2.resize(face_img, (256, 256))
                            # body_resized = cv2.resize(body_img, (256, 256))
                            target_size = (256, 256)
                            
                            # 保持宽高比缩放并填充人脸图像
                            h, w, _ = face_img.shape
                            scale = min(target_size[0] / w, target_size[1] / h)
                            new_w, new_h = int(w * scale), int(h * scale)
                            face_resized = cv2.resize(face_img, (new_w, new_h))
                            face_padded = np.full((target_size[1], target_size[0], 3), 0, dtype=np.uint8) # 黑色填充
                            face_padded[(target_size[1] - new_h) // 2 : (target_size[1] - new_h) // 2 + new_h,
                                        (target_size[0] - new_w) // 2 : (target_size[0] - new_w) // 2 + new_w] = face_resized
                            
                            # 保持宽高比缩放并填充身体图像
                            h, w, _ = body_img.shape
                            scale = min(target_size[0] / w, target_size[1] / h)
                            new_w, new_h = int(w * scale), int(h * scale)
                            body_resized = cv2.resize(body_img, (new_w, new_h))
                            body_padded = np.full((target_size[1], target_size[0], 3), 0, dtype=np.uint8) # 黑色填充
                            body_padded[(target_size[1] - new_h) // 2 : (target_size[1] - new_h) // 2 + new_h,
                                        (target_size[0] - new_w) // 2 : (target_size[0] - new_w) // 2 + new_w] = body_resized
                            
                            return face_padded, body_padded  # 返回裁剪后的图像数据
        
        return None, None
    
    except Exception as e:
        # print(f"处理帧时出错: {str(e)}") # 调试时可以取消注释
        return None, None

def main():
    """
    主函数
    """
    print("开始处理Celeb-DF-v2数据集...")
    print("="*50)
    
    # 检查输入路径是否存在
    dataset_path = r"D:\Dataset\Celeb-DF-v2"
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请确保Celeb-DF-v2数据集已下载到指定路径")
        return
    
    try:
        process_celeb_df_dataset()
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
    
    print("\n程序结束")

if __name__ == "__main__":
    main()