# quick_test.py - Quick test with your model

import cv2
import numpy as np
from ultralytics import YOLO

def quick_test():
    # Load model
    print("Loading model...")
    model = YOLO("models/best_homeobjects.pt")
    print(f"Model loaded with {len(model.names)} classes")
    
    # Test on a video if available
    video_path = "/Users/nomeshgaur/Desktop/AeroGuard-MTP1/test_data/video.mp4"  # Change this to your video path
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("Please place a test video in this folder")
        return
    
    print(f"Testing on: {video_path}")
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_conf = []
    total_detections = 0
    
    while frame_count < 30:  # Process first 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.25)
        detections = results[0].boxes
        
        if len(detections) > 0:
            confs = detections.conf.cpu().numpy()
            total_conf.extend(confs)
            total_detections += len(detections)
        
        frame_count += 1
        
        # Show progress
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    
    # Show results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    
    if total_conf:
        print(f"Average confidence: {np.mean(total_conf):.2%}")
        print(f"Confidence range: {np.min(total_conf):.2%} - {np.max(total_conf):.2%}")
    else:
        print("⚠️ No objects detected in any frame!")

if __name__ == "__main__":
    import os
    quick_test()