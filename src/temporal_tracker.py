import numpy as np

class TemporalTracker:
    def __init__(self, iou_threshold=0.5, max_age=5):
        self.iou_thresh = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 0

    def _iou(self, box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        # detections: list of dicts with keys 'class_name', 'bbox', 'confidence'
        matched = []
        unmatched_dets = list(range(len(detections)))
        for track in self.tracks:
            best_iou = 0
            best_idx = -1
            for d_idx in unmatched_dets:
                iou = self._iou(track['last_bbox'], detections[d_idx]['bbox'])
                if iou > best_iou and iou > self.iou_thresh:
                    best_iou = iou
                    best_idx = d_idx
            if best_idx >= 0:
                matched.append((track, detections[best_idx]))
                unmatched_dets.remove(best_idx)
            else:
                track['age'] += 1

        for d_idx in unmatched_dets:
            new_track = {
                'id': self.next_id,
                'class': detections[d_idx]['class_name'],
                'bbox_history': [detections[d_idx]['bbox']],
                'conf_history': [detections[d_idx]['confidence']],
                'last_bbox': detections[d_idx]['bbox'],
                'age': 0
            }
            self.tracks.append(new_track)
            self.next_id += 1

        for track, det in matched:
            track['bbox_history'].append(det['bbox'])
            track['conf_history'].append(det['confidence'])
            track['last_bbox'] = det['bbox']
            track['age'] = 0
            if len(track['bbox_history']) > 10:
                track['bbox_history'].pop(0)
                track['conf_history'].pop(0)

        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # Return smoothed detections
        results = []
        for track in self.tracks:
            bboxes = np.array(track['bbox_history'])
            median_bbox = np.median(bboxes, axis=0).astype(int)
            width_px = median_bbox[2] - median_bbox[0]
            height_px = median_bbox[3] - median_bbox[1]
            results.append({
                'class_name': track['class'],
                'width_px': width_px,
                'height_px': height_px,
                'confidence': np.mean(track['conf_history'])
            })
        return results