import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="models/last.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_np):
        """Run detection on a single image (BGR or RGB)."""
        results = self.model(image_np, conf=self.conf_threshold)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            detections.append({
                'class_name': self.model.names[int(box.cls[0])],
                'bbox': (x1, y1, x2, y2),
                'confidence': float(box.conf[0])
            })
        return detections, results[0].plot()
    
model = YOLO("models/best.pt")  # or yolov8n.pt

print("\n best module list best.pt ", model.names)

model = YOLO("models/last.pt")  # or yolov8n.pt

print("\n last module list last.pt: ",model.names)

model = YOLO("models/yolov8n.pt")  # or yolov8n.pt

print("\n Yolo module list yolov8n ",model.names)

model = YOLO("models/best_homeobjects.pt")  # or yolov8n.pt

print("\n best module list homeobjects ",model.names)