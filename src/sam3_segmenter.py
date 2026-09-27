# src/sam3_segmenter.py
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

class SAM3Segmenter:
    """
    SAM3 for concept-based segmentation of indoor objects.
    """
    
    def __init__(self, model_path: str = "sam3_b.pt", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load SAM3 model."""
        # Using the official SAM3 implementation
        # Note: SAM3 checkpoints are available from Meta
        try:
            from sam3.model import SAM3
            model = SAM3.from_pretrained(model_path)
            model.to(self.device)
            return model
        except ImportError:
            # Fallback to SAM2
            print("⚠️ SAM3 not available. Using SAM2 fallback.")
            return self.load_sam2()
    
    def load_sam2(self):
        """Fallback to SAM2."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        model_cfg = "sam2_hiera_large.yaml"
        model = build_sam2(model_cfg)
        predictor = SAM2ImagePredictor(model)
        return predictor
    
    def segment_objects(self, image: np.ndarray, concept: str = None) -> List[Dict]:
        """
        Segment objects in image based on concept prompt.
        
        Args:
            image: RGB image as numpy array
            concept: Text prompt (e.g., "bed", "window", "door")
            
        Returns:
            List of dicts with masks, bboxes, and class labels
        """
        if hasattr(self.model, 'predict'):
            # SAM3
            if concept:
                result = self.model.predict_concept(image, concept)
            else:
                result = self.model.predict_all(image)
            
            return self.parse_sam3_output(result)
        else:
            # SAM2 fallback
            return self.segment_with_sam2(image, concept)
    
    def parse_sam3_output(self, result) -> List[Dict]:
        """Parse SAM3 output into standard format."""
        objects = []
        
        # SAM3 returns masks, scores, and class labels
        masks = result.get('masks', [])
        scores = result.get('scores', [])
        labels = result.get('labels', [])
        
        for mask, score, label in zip(masks, scores, labels):
            if score > 0.5:  # Confidence threshold
                # Convert mask to bounding box
                y, x = np.where(mask)
                if len(y) > 0:
                    bbox = [min(x), min(y), max(x), max(y)]
                    objects.append({
                        'mask': mask,
                        'bbox': bbox,
                        'score': float(score),
                        'label': label
                    })
        
        return objects
    
    def segment_with_sam2(self, image: np.ndarray, concept: str) -> List[Dict]:
        """Segment with SAM2 (fallback)."""
        # SAM2 requires prompts or automatic segmentation
        # Use YOLO to get proposals, then refine with SAM2
        from ultralytics import YOLO
        yolo = YOLO('yolov8n.pt')
        detections = yolo(image, conf=0.25)
        
        results = []
        self.model.set_image(image)
        
        for box in detections[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            mask, score, _ = self.model.predict(box=bbox)
            
            results.append({
                'mask': mask,
                'bbox': bbox,
                'score': float(score),
                'label': self.model.names[int(box.cls)]
            })
        
        return results
    
    def segment_video(self, video_frames: List[np.ndarray], concept: str) -> Dict:
        """Segment objects across video frames."""
        tracked_objects = {}
        
        for i, frame in enumerate(video_frames):
            objects = self.segment_objects(frame, concept)
            
            # Track objects across frames
            for obj in objects:
                key = f"{obj['label']}_{obj['bbox']}"
                if key not in tracked_objects:
                    tracked_objects[key] = {
                        'label': obj['label'],
                        'masks': [],
                        'scores': [],
                        'frames': []
                    }
                tracked_objects[key]['masks'].append(obj['mask'])
                tracked_objects[key]['scores'].append(obj['score'])
                tracked_objects[key]['frames'].append(i)
        
        # Aggregate tracking results
        result = {}
        for key, data in tracked_objects.items():
            if len(data['masks']) > 2:  # Seen in at least 3 frames
                result[key] = {
                    'label': data['label'],
                    'masks': data['masks'],
                    'avg_score': np.mean(data['scores']),
                    'frames': data['frames']
                }
        
        return result