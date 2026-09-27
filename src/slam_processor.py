# src/slam_processor.py
import cv2
import numpy as np
from pathlib import Path
import json

class SLAMProcessor:
    """
    Process video with SLAM to get camera poses and point cloud.
    Uses OpenCV-based SLAM (lightweight, no external dependencies).
    """
    
    def __init__(self, focal_length=800, feature_match_ratio=0.75):
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = feature_match_ratio
        self.focal_length = focal_length
        
        # Tracking state
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.camera_poses = []      # 4x4 matrices
        self.keypoints_3d = []      # 3D points triangulated
        self.keyframes = []         # (frame, pose, index)
        self.K = None               # camera intrinsic matrix
        
    def set_camera_intrinsics(self, width, height):
        """Set camera intrinsic matrix (approximate)."""
        fx = fy = self.focal_length
        cx = width / 2
        cy = height / 2
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)
        return self.K
    
    def process_video(self, video_path, frame_interval=5):
        """
        Process video and return SLAM results.
        
        Args:
            video_path: Path to video file
            frame_interval: Process every Nth frame
            
        Returns:
            dict: {
                'camera_poses': list of 4x4 matrices,
                'point_cloud': list of 3D points,
                'keyframes': list of (frame_idx, image, pose),
                'scale': estimated scale factor,
                'detection_frames': list of frames for object detection
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_camera_intrinsics(width, height)
        
        frame_count = 0
        processed_frames = 0
        keyframe_indices = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process frame for SLAM
            if frame_count % frame_interval == 0:
                pose = self.process_frame(gray)
                if pose is not None and len(self.camera_poses) % 5 == 0:
                    # Store as keyframe for object detection
                    self.keyframes.append({
                        'frame_idx': frame_count,
                        'image': frame.copy(),
                        'pose': pose.copy(),
                        'gray': gray
                    })
                    keyframe_indices.append(frame_count)
                
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        # Estimate scale
        scale = self.estimate_scale()
        
        # Select frames for object detection (every 10th keyframe)
        detection_frames = [kf['image'] for kf in self.keyframes[::2]]
        detection_poses = [kf['pose'] for kf in self.keyframes[::2]]
        detection_indices = [kf['frame_idx'] for kf in self.keyframes[::2]]
        
        return {
            'camera_poses': self.camera_poses,
            'point_cloud': self.keypoints_3d,
            'keyframes': self.keyframes,
            'scale': scale,
            'detection_frames': detection_frames,
            'detection_poses': detection_poses,
            'detection_indices': detection_indices,
            'camera_matrix': self.K,
            'image_size': (width, height)
        }
    
    def process_frame(self, gray_frame):
        """Process a single frame for SLAM."""
        if self.K is None:
            raise ValueError("Camera intrinsics not set. Call set_camera_intrinsics() first.")
        
        kp, des = self.orb.detectAndCompute(gray_frame, None)
        if kp is None or len(kp) < 10:
            return None
        
        if self.prev_frame is None:
            # First frame - initialize
            self.prev_frame = gray_frame
            self.prev_kp = kp
            self.prev_des = des
            self.camera_poses.append(np.eye(4))
            return np.eye(4)
        
        # Match features
        matches = self.bf.knnMatch(self.prev_des, des, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 15:
            # Not enough matches - reset tracking
            self.prev_frame = gray_frame
            self.prev_kp = kp
            self.prev_des = des
            return None
        
        # Get matched points
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches])
        
        # Find essential matrix and recover pose
        E, mask = cv2.findEssentialMat(pts_prev, pts_curr, self.K, method=cv2.RANSAC)
        if E is None:
            return None
        
        _, R, t, mask_pose = cv2.recoverPose(E, pts_prev, pts_curr, self.K)
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        # Update camera pose
        current_pose = self.camera_poses[-1] @ T
        self.camera_poses.append(current_pose)
        
        # Triangulate 3D points (for point cloud)
        P1 = self.K @ self.camera_poses[-2][:3, :]
        P2 = self.K @ current_pose[:3, :]
        
        points_4d = cv2.triangulatePoints(P1, P2, pts_prev.T, pts_curr.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        self.keypoints_3d.extend(points_3d)
        
        # Update for next frame
        self.prev_frame = gray_frame
        self.prev_kp = kp
        self.prev_des = des
        
        return current_pose
    
    def estimate_scale(self):
        """Estimate scale factor from camera motion."""
        if len(self.camera_poses) < 2:
            return 1.0
        
        # Total camera movement
        total_movement = 0.0
        for i in range(1, len(self.camera_poses)):
            trans = self.camera_poses[i][:3, 3] - self.camera_poses[i-1][:3, 3]
            total_movement += np.linalg.norm(trans)
        
        # Assume walking speed ~0.5 m/s, video length ~ frames/30
        # This gives approximate scale
        avg_speed = 50.0  # cm/s
        video_duration = len(self.camera_poses) / 30.0  # seconds
        expected_movement = avg_speed * video_duration  # cm
        
        if total_movement > 0:
            scale = expected_movement / total_movement
        else:
            scale = 1.0
        
        return scale


class Object3DProjector:
    """
    Project 2D object detections into 3D space using SLAM results.
    """
    
    def __init__(self, camera_matrix, image_size):
        self.K = camera_matrix
        self.width, self.height = image_size
        
    def deproject_bbox(self, bbox, depth_map, camera_pose):
        """
        Deproject a 2D bounding box to 3D space.
        
        Args:
            bbox: (x1, y1, x2, y2) in pixels
            depth_map: 2D array of depths (from SLAM or depth model)
            camera_pose: 4x4 camera pose matrix
            
        Returns:
            dict: {
                'center_3d': (x, y, z) in world coordinates,
                'width': real width,
                'height': real height,
                'depth': real depth
            }
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Get depth at centre of bounding box
        if depth_map is not None:
            depth = depth_map[int(cy), int(cx)]
        else:
            # Use average depth of bounding box region
            region = depth_map[int(y1):int(y2), int(x1):int(x2)]
            depth = np.median(region) if region.size > 0 else 1.0
        
        # Convert pixel to camera coordinates
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        u0 = self.K[0, 2]
        v0 = self.K[1, 2]
        
        X_c = (cx - u0) * depth / fx
        Y_c = (cy - v0) * depth / fy
        Z_c = depth
        
        # Convert to world coordinates
        point_cam = np.array([X_c, Y_c, Z_c, 1.0])
        point_world = camera_pose @ point_cam
        
        # Compute 3D bounding box dimensions
        # Use depth at corners to get real width/height
        corners = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2)
        ]
        depths_corners = []
        for px, py in corners:
            d = depth_map[int(py), int(px)] if depth_map is not None else depth
            depths_corners.append(d)
        
        # Estimate width and height in 3D
        # Use difference in X and Y directions
        depth_center = depth
        width_3d = (x2 - x1) * depth_center / fx
        height_3d = (y2 - y1) * depth_center / fy
        depth_3d = width_3d * 0.8  # approximate
        
        return {
            'center': point_world[:3],
            'width': width_3d,
            'height': height_3d,
            'depth': depth_3d,
            'position': point_world[:3]
        }
    
    def project_detections_to_3d(self, detections, depth_map, camera_pose):
        """
        Project all detections to 3D space.
        
        Args:
            detections: list of detection dicts with 'bbox', 'class_name', 'confidence'
            depth_map: 2D depth array
            camera_pose: 4x4 camera pose matrix
            
        Returns:
            list of 3D detections with 3D positions and dimensions
        """
        results = []
        for det in detections:
            try:
                det_3d = self.deproject_bbox(
                    det['bbox'], depth_map, camera_pose
                )
                det_3d['class_name'] = det['class_name']
                det_3d['confidence'] = det['confidence']
                results.append(det_3d)
            except Exception as e:
                # Skip failed projections
                continue
        return results