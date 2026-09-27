# test_detection.py - Updated version with test_data folder

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from collections import defaultdict

def test_current_model():
    """Test your current YOLO model on sample images/videos"""
    
    print("=" * 60)
    print("🔍 TESTING YOUR CURRENT OBJECT DETECTION")
    print("=" * 60)
    
    # 1. Load your model
    print("\n📂 Loading your current model...")
    
    # Try your fine-tuned model first
    model_path = "models/best_homeobjects.pt"
    if os.path.exists(model_path):
        print(f"✅ Found fine-tuned model: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"⚠️ Fine-tuned model not found, using default YOLOv8n")
        model = YOLO("yolov8n.pt")
    
    # 2. Check model info
    print("\n📊 Model Information:")
    print(f"   Classes: {len(model.names)}")
    print(f"   Class names: {list(model.names.values())[:10]}...")  # Show first 10
    
    # 3. Find test data - Check multiple locations
    test_data_folders = ['.', 'test_data', 'data', 'samples', 'test_images']
    test_images = []
    test_videos = []
    
    for folder in test_data_folders:
        if os.path.exists(folder):
            # Look for images
            for ext in ['jpg', 'jpeg', 'png']:
                test_images.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(f'.{ext}')])
            # Look for videos
            for ext in ['mp4', 'mov', 'avi']:
                test_videos.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(f'.{ext}')])
    
    # 4. Test on images
    print("\n" + "=" * 60)
    print("📸 TESTING ON IMAGE")
    print("=" * 60)
    
    if test_images:
        print(f"\n📷 Found {len(test_images)} test images")
        for img_path in test_images[:3]:  # Test up to 3 images
            print(f"\n--- Testing: {os.path.basename(img_path)} ---")
            test_image_detection(model, img_path)
    else:
        print("\n⚠️ No test images found!")
        print(f"   Checked these folders: {test_data_folders}")
        print("   Please place test images in one of these folders.")
    
    # 5. Test on videos
    print("\n" + "=" * 60)
    print("🎬 TESTING ON VIDEO")
    print("=" * 60)
    
    if test_videos:
        print(f"\n🎥 Found {len(test_videos)} test videos")
        for vid_path in test_videos[:1]:  # Test first video only
            print(f"\n--- Testing: {os.path.basename(vid_path)} ---")
            test_video_detection(model, vid_path)
    else:
        print("\n⚠️ No test videos found!")
        print(f"   Checked these folders: {test_data_folders}")
        print("   Please place test videos in one of these folders.")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)

# ... rest of the functions remain the same ...
    test_videos = []
    for ext in ['mp4', 'mov', 'avi']:
        test_videos.extend([f for f in os.listdir('.') if f.endswith(f'.{ext}')])
    
    if test_videos:
        test_vid = test_videos[0]
        print(f"\n🎥 Testing on: {test_vid}")
        test_video_detection(model, test_vid)
    else:
        print("\n⚠️ No test video found. Upload a sample video to test.")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)

def test_image_detection(model, image_path):
    """Test detection on a single image"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image")
        return
    
    print(f"   Image size: {img.shape}")
    
    # Run detection
    print("\n🔄 Running detection...")
    start_time = time.time()
    results = model(img, conf=0.25)
    inference_time = time.time() - start_time
    
    # Analyze results
    detections = results[0].boxes
    print(f"\n📊 Results:")
    print(f"   Inference time: {inference_time:.3f} seconds")
    print(f"   Number of detections: {len(detections)}")
    
    if len(detections) > 0:
        # Get confidence scores
        confs = detections.conf.cpu().numpy()
        classes = detections.cls.cpu().numpy()
        
        print(f"\n   Confidence Statistics:")
        print(f"   - Average: {confs.mean():.2%}")
        print(f"   - Minimum: {confs.min():.2%}")
        print(f"   - Maximum: {confs.max():.2%}")
        
        # Show detections by class
        class_counts = defaultdict(int)
        class_confs = defaultdict(list)
        
        for conf, cls in zip(confs, classes):
            class_name = model.names[int(cls)]
            class_counts[class_name] += 1
            class_confs[class_name].append(conf)
        
        print(f"\n   Detected Objects:")
        for cls, count in class_counts.items():
            avg_conf = np.mean(class_confs[cls])
            print(f"   - {cls}: {count} detections (avg conf: {avg_conf:.2%})")
        
        # Identify low confidence detections
        low_conf = np.where(confs < 0.3)[0]
        if len(low_conf) > 0:
            print(f"\n   ⚠️ {len(low_conf)} detections have low confidence (<30%)")
            for idx in low_conf:
                class_name = model.names[int(classes[idx])]
                print(f"      - {class_name}: {confs[idx]:.2%}")
        
        # Overall assessment
        print(f"\n   📈 Performance Assessment:")
        if confs.mean() > 0.7:
            print(f"   ✅ Excellent average confidence!")
        elif confs.mean() > 0.5:
            print(f"   👍 Good average confidence, but could be improved")
        else:
            print(f"   ⚠️ Low average confidence - need improvement")
            
        if len(low_conf) > len(detections) * 0.3:
            print(f"   ⚠️ Many low-confidence detections - model may need improvement")
            
    else:
        print("\n   ⚠️ No objects detected!")
        print("   This could mean:")
        print("   - The image doesn't contain any objects your model knows")
        print(f"   - Your model was trained on {len(model.names)} classes: {list(model.names.values())[:5]}...")
        print("   - Try lowering the confidence threshold")
    
    return results

def test_video_detection(model, video_path):
    """Test detection on a video (first 50 frames)"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Video: {total_frames} frames at {fps:.2f} FPS")
    
    # Process first 50 frames or until end
    max_frames = min(50, total_frames)
    print(f"\n🔄 Processing {max_frames} frames...")
    
    frame_times = []
    total_detections = 0
    total_conf = []
    frames_with_detections = 0
    
    for frame_num in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        start_time = time.time()
        results = model(frame, conf=0.25)
        inference_time = time.time() - start_time
        frame_times.append(inference_time)
        
        # Count detections
        detections = results[0].boxes
        if len(detections) > 0:
            frames_with_detections += 1
            total_detections += len(detections)
            confs = detections.conf.cpu().numpy()
            total_conf.extend(confs)
    
    cap.release()
    
    # Analyze video results
    print(f"\n📊 Results:")
    print(f"   Frames processed: {max_frames}")
    print(f"   Frames with detections: {frames_with_detections} ({frames_with_detections/max_frames:.1%})")
    print(f"   Total detections: {total_detections}")
    print(f"   Average detections per frame: {total_detections/max_frames:.2f}")
    print(f"   Average inference time: {np.mean(frame_times):.3f}s per frame")
    
    if total_conf:
        print(f"\n   Confidence Statistics:")
        print(f"   - Average: {np.mean(total_conf):.2%}")
        print(f"   - Minimum: {np.min(total_conf):.2%}")
        print(f"   - Maximum: {np.max(total_conf):.2%}")
    
    # Performance assessment
    print(f"\n   📈 Performance Assessment:")
    if frames_with_detections/max_frames > 0.8:
        print(f"   ✅ Model detects objects in most frames")
    else:
        print(f"   ⚠️ Model misses objects in many frames")
    
    if total_conf and np.mean(total_conf) > 0.7:
        print(f"   ✅ High confidence detections")
    elif total_conf and np.mean(total_conf) > 0.5:
        print(f"   👍 Acceptable confidence, room for improvement")
    else:
        print(f"   ⚠️ Low confidence detections - need improvement")
    
    return results

if __name__ == "__main__":
    test_current_model()