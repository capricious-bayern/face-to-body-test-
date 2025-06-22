import torch
from facenet_pytorch import MTCNN
import cv2
from PIL import Image

class FaceDetector:
    def __init__(self, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.mtcnn = MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            keep_all=True, # Keep all detected faces
            device=self.device
        )

    def detect(self, image_path):
        """
        检测图像中的人脸并返回裁剪后的人脸图像和边界框。
        Args:
            image_path (str): 输入图像的路径。
        Returns:
            tuple: (list of PIL.Image.Image, list of list), 裁剪后的人脸图像列表和边界框列表。
        """
        img = Image.open(image_path).convert('RGB')
        
        boxes, probs = self.mtcnn.detect(img)
        
        cropped_faces = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                # 确保边界框坐标是整数
                box = [int(b) for b in box]
                x1, y1, x2, y2 = box
                
                # 裁剪人脸
                cropped_face = img.crop((x1, y1, x2, y2))
                cropped_faces.append(cropped_face)
                
        return cropped_faces, boxes

if __name__ == "__main__":
    # 示例用法
    # 创建一个虚拟图像文件用于测试
    from PIL import ImageDraw
    dummy_image_path = "./dummy_face_test.jpg"
    dummy_image = Image.new('RGB', (600, 400), color = 'blue')
    draw = ImageDraw.Draw(dummy_image)
    draw.ellipse((100, 50, 300, 250), fill='yellow', outline='black') # 模拟一个脸
    dummy_image.save(dummy_image_path)

    face_detector = FaceDetector()
    cropped_faces, boxes = face_detector.detect(dummy_image_path)

    if cropped_faces:
        print(f"Detected {len(cropped_faces)} face(s).")
        for i, face in enumerate(cropped_faces):
            face.save(f"./detected_face_{i}.jpg")
            print(f"Saved detected_face_{i}.jpg")
    else:
        print("No faces detected.")

    # 清理虚拟文件
    os.remove(dummy_image_path)
    for i in range(len(cropped_faces)):
        os.remove(f"./detected_face_{i}.jpg")