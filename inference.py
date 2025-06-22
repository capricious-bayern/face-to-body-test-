import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import argparse
import yaml
from typing import Tuple, Optional, Dict, List
import time
from pathlib import Path

# 导入自定义模块
from models.face_encoder import create_face_encoder
from models.body_encoder import create_body_encoder
from models.classifier import create_classifier
from process_celeb_df import process_frame  # 复用数据处理逻辑
from facenet_pytorch import MTCNN
from ultralytics import YOLO

class FaceBodyInference:
    """
    人脸-身体匹配检测推理器
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
        
        print(f"Using device: {self.device}")
        
        # 初始化检测模型
        self._init_detection_models()
        
        # 加载训练好的模型
        self._load_models(checkpoint_path)
        
        # 设置为评估模式
        self.face_encoder.eval()
        self.body_encoder.eval()
        self.classifier.eval()
        
        print("Models loaded and ready for inference")
    
    def _init_detection_models(self):
        """初始化检测模型"""
        self.mtcnn = MTCNN(keep_all=True, min_face_size=20, device=self.device)
        self.yolo_model = YOLO('yolov8n.pt')
        
    def _load_models(self, checkpoint_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 创建模型
        self.face_encoder = create_face_encoder(
            model_type=self.config['model']['face_encoder']['type'],
            **self.config['model']['face_encoder']['params']
        ).to(self.device)
        
        self.body_encoder = create_body_encoder(
            model_type=self.config['model']['body_encoder']['type'],
            **self.config['model']['body_encoder']['params']
        ).to(self.device)
        
        self.classifier = create_classifier(
            classifier_type=self.config['model']['classifier']['type'],
            **self.config['model']['classifier']['params']
        ).to(self.device)
        
        # 加载权重
        self.face_encoder.load_state_dict(checkpoint['face_encoder_state_dict'])
        self.body_encoder.load_state_dict(checkpoint['body_encoder_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
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
    
    def predict_single_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        对单帧进行预测
        
        Args:
            frame: 输入帧
            
        Returns:
            result: 预测结果字典
        """
        start_time = time.time()
        
        # 预处理
        face_tensor, body_tensor = self.preprocess_frame(frame)
        
        if face_tensor is None or body_tensor is None:
            return {
                'success': False,
                'message': 'Failed to detect face or body',
                'processing_time': time.time() - start_time
            }
        
        # 推理
        with torch.no_grad():
            # 提取特征
            face_features = self.face_encoder.get_features(face_tensor)
            body_features = self.body_encoder.get_features(body_tensor)
            
            # 分类
            logits = self.classifier(face_features, body_features)
            probabilities = F.softmax(logits, dim=1)
            
            # 预测结果
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, prediction].item()
            
            # 各类别概率
            fake_prob = probabilities[0, 0].item()  # 伪造概率
            real_prob = probabilities[0, 1].item()  # 真实概率
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'prediction': 'real' if prediction == 1 else 'fake',
            'confidence': confidence,
            'probabilities': {
                'fake': fake_prob,
                'real': real_prob
            },
            'processing_time': processing_time
        }
    
    def predict_video(self, video_path: str, frame_interval: int = 30, 
                     output_path: Optional[str] = None) -> Dict[str, any]:
        """
        对视频进行预测
        
        Args:
            video_path: 视频文件路径
            frame_interval: 帧间隔
            output_path: 输出结果文件路径
            
        Returns:
            results: 预测结果
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'success': False, 'message': f'Cannot open video: {video_path}'}
        
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, Processing every {frame_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔frame_interval帧处理一次
            if frame_count % frame_interval == 0:
                result = self.predict_single_frame(frame)
                result['frame_number'] = frame_count
                frame_results.append(result)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        # 统计结果
        successful_predictions = [r for r in frame_results if r['success']]
        
        if not successful_predictions:
            return {
                'success': False,
                'message': 'No successful predictions',
                'total_frames': frame_count,
                'processed_frames': processed_count
            }
        
        # 计算统计信息
        fake_count = sum(1 for r in successful_predictions if r['prediction'] == 'fake')
        real_count = sum(1 for r in successful_predictions if r['prediction'] == 'real')
        
        avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
        avg_fake_prob = np.mean([r['probabilities']['fake'] for r in successful_predictions])
        avg_real_prob = np.mean([r['probabilities']['real'] for r in successful_predictions])
        avg_processing_time = np.mean([r['processing_time'] for r in successful_predictions])
        
        # 整体预测（基于多数投票）
        overall_prediction = 'fake' if fake_count > real_count else 'real'
        overall_confidence = max(fake_count, real_count) / len(successful_predictions)
        
        results = {
            'success': True,
            'video_path': video_path,
            'overall_prediction': overall_prediction,
            'overall_confidence': overall_confidence,
            'statistics': {
                'total_frames': frame_count,
                'processed_frames': processed_count,
                'successful_predictions': len(successful_predictions),
                'fake_predictions': fake_count,
                'real_predictions': real_count,
                'avg_confidence': avg_confidence,
                'avg_fake_probability': avg_fake_prob,
                'avg_real_probability': avg_real_prob,
                'avg_processing_time': avg_processing_time
            },
            'frame_results': frame_results
        }
        
        # 保存结果
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def predict_batch_videos(self, video_dir: str, output_dir: str, 
                           frame_interval: int = 30) -> List[Dict]:
        """
        批量处理视频
        
        Args:
            video_dir: 视频目录
            output_dir: 输出目录
            frame_interval: 帧间隔
            
        Returns:
            all_results: 所有视频的预测结果
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有视频文件
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
        
        all_results = []
        
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            
            # 预测
            result = self.predict_video(
                str(video_file), 
                frame_interval=frame_interval,
                output_path=str(output_dir / f"{video_file.stem}_result.json")
            )
            
            all_results.append(result)
            
            if result['success']:
                print(f"Result: {result['overall_prediction']} (confidence: {result['overall_confidence']:.3f})")
            else:
                print(f"Failed: {result['message']}")
        
        # 保存汇总结果
        summary_path = output_dir / 'batch_summary.json'
        self._save_results({
            'total_videos': len(video_files),
            'successful_predictions': sum(1 for r in all_results if r['success']),
            'results': all_results
        }, str(summary_path))
        
        return all_results
    
    def _save_results(self, results: Dict, output_path: str):
        """保存结果到JSON文件"""
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        # 递归转换
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = recursive_convert(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Face-Body Matching Inference')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True, help='Input video file or directory')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--frame_interval', type=int, default=30, help='Frame sampling interval')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device type')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = FaceBodyInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单个视频文件
        output_path = args.output if args.output else f"{input_path.stem}_result.json"
        result = inferencer.predict_video(
            str(input_path),
            frame_interval=args.frame_interval,
            output_path=output_path
        )
        
        if result['success']:
            print(f"\nPrediction: {result['overall_prediction']}")
            print(f"Confidence: {result['overall_confidence']:.3f}")
            print(f"Processed {result['statistics']['successful_predictions']} frames")
        else:
            print(f"\nPrediction failed: {result['message']}")
    
    elif input_path.is_dir():
        # 批量处理
        output_dir = args.output if args.output else 'inference_results'
        results = inferencer.predict_batch_videos(
            str(input_path),
            output_dir,
            frame_interval=args.frame_interval
        )
        
        # 打印汇总
        successful = sum(1 for r in results if r['success'])
        print(f"\nBatch processing completed:")
        print(f"Total videos: {len(results)}")
        print(f"Successful predictions: {successful}")
        
        if successful > 0:
            fake_count = sum(1 for r in results if r['success'] and r['overall_prediction'] == 'fake')
            real_count = successful - fake_count
            print(f"Fake videos detected: {fake_count}")
            print(f"Real videos detected: {real_count}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == '__main__':
    main()