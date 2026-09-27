# quick_test_v9.py - Test with YOLOv9

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_yolov9():
    """Test YOLOv9 on your video"""
    
    print("📥 Downloading/loading YOLOv9c...")
    model = YOLO("yolov9c.pt")  # Will auto-download
    
    print(f"Model loaded with {len(model.names)} classes")
    print(f"Classes: {list(model.names.values())[:10]}...")
    
    # Test on video
    video_path = "test_data/video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_conf = []
    
    print(f"\nTesting on: {video_path}")
    
    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.25)
        detections = results[0].boxes
        
        if len(detections) > 0:
            confs = detections.conf.cpu().numpy()
            total_conf.extend(confs)
        
        frame_count += 1
    
    cap.release()
    
    # Results
    print("\n" + "="*50)
    print("📊 RESULTS WITH YOLOv9")
    print("="*50)
    print(f"Frames: {frame_count}")
    print(f"Total detections: {len(total_conf)}")
    
    if total_conf:
        print(f"Average confidence: {np.mean(total_conf):.2%}")
        print(f"Confidence range: {np.min(total_conf):.2%} - {np.max(total_conf):.2%}")
    else:
        print("No detections!")

if __name__ == "__main__":
    test_yolov9()