 # sam_test.py - Test with SAM enabled

import cv2
import numpy as np
from ultralytics import YOLO, SAM
import os

def test_with_sam():
    """Test detection with SAM refinement"""
    
    # Load models
    print("Loading models...")
    yolo = YOLO("models/best_homeobjects.pt")
    
    try:
        sam = SAM("sam_b.pt")
        print("✅ SAM loaded successfully")
    except:
        print("⚠️ SAM not available")
        sam = None
    
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
        
        # YOLO detection
        results = yolo(frame, conf=0.25)
        detections = results[0].boxes
        
        if len(detections) > 0:
            # Get confidence
            confs = detections.conf.cpu().numpy()
            total_conf.extend(confs)
            
            # If SAM available, refine detections
            if sam:
                bboxes = [box.xyxy[0].cpu().numpy() for box in detections]
                try:
                    sam_results = sam(frame, bboxes=bboxes)
                    print(f"  SAM refined {len(detections)} detections")
                except:
                    pass
        
        frame_count += 1
    
    cap.release()
    
    # Results
    print("\n" + "="*50)
    print("📊 RESULTS WITH SAM")
    print("="*50)
    print(f"Frames: {frame_count}")
    print(f"Total detections: {len(total_conf)}")
    
    if total_conf:
        print(f"Average confidence: {np.mean(total_conf):.2%}")
        print(f"Confidence range: {np.min(total_conf):.2%} - {np.max(total_conf):.2%}")
    else:
        print("No detections!")

if __name__ == "__main__":
    test_with_sam()