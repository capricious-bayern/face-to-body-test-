import torch
from ultralytics import YOLO
import cv2
from PIL import Image

class BodyDetector:
    def __init__(self, model_path='yolov8n.pt', device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image_path, confidence_threshold=0.7):
        """
        检测图像中的人体并返回裁剪后的人体图像和边界框。
        Args:
            image_path (str): 输入图像的路径。
            confidence_threshold (float): 检测的最小置信度。
        Returns:
            tuple: (list of PIL.Image.Image, list of list), 裁剪后的人体图像列表和边界框列表。
        """
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        results = self.model(img_cv2, verbose=False) # verbose=False to suppress output
        
        cropped_bodies = []
        boxes_list = []

        for r in results:
            for box in r.boxes:
                # class 0 is 'person' in COCO dataset
                if int(box.cls[0]) == 0 and box.conf[0] >= confidence_threshold:
                    x1, y1, x2, y2 = [int(b) for b in box.xyxy[0]]
                    
                    # 确保裁剪区域有效
                    if x1 < x2 and y1 < y2:
                        cropped_body_cv2 = img_cv2[y1:y2, x1:x2]
                        if cropped_body_cv2.size > 0: # 检查裁剪后的图像是否为空
                            cropped_bodies.append(Image.fromarray(cv2.cvtColor(cropped_body_cv2, cv2.COLOR_BGR2RGB)))
                            boxes_list.append([x1, y1, x2, y2])
                
        return cropped_bodies, boxes_list

if __name__ == "__main__":
    # 示例用法
    # 创建一个虚拟图像文件用于测试
    from PIL import ImageDraw
    dummy_image_path = "./dummy_body_test.jpg"
    dummy_image = Image.new('RGB', (800, 600), color = 'green')
    draw = ImageDraw.Draw(dummy_image)
    draw.rectangle((100, 50, 400, 550), fill='purple', outline='black') # 模拟一个人形
    dummy_image.save(dummy_image_path)

    body_detector = BodyDetector()
    cropped_bodies, boxes = body_detector.detect(dummy_image_path)

    if cropped_bodies:
        print(f"Detected {len(cropped_bodies)} body(ies).")
        for i, body in enumerate(cropped_bodies):
            body.save(f"./detected_body_{i}.jpg")
            print(f"Saved detected_body_{i}.jpg")
    else:
        print("No bodies detected.")

    # 清理虚拟文件
    os.remove(dummy_image_path)
    for i in range(len(cropped_bodies)):
        os.remove(f"./detected_body_{i}.jpg")