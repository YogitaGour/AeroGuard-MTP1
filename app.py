# app.py – Complete AeroGuard with SLAM + 3D Visualization (Fixed Version)

import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO, SAM
from src.slam_processor import SLAMProcessor, Object3DProjector
from src.dust_simulator import DustSimulator
from src.visualizer import plot_risk_heatmap
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

st.set_page_config(page_title="AeroGuard – 3D Digital Twin", layout="wide")
st.title("🛡️ AeroGuard – 3D Digital Twin with SLAM")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    """Load YOLO and SAM models."""
    models = {}
    
    # Try new trained model first
    model_paths = [
        "runs/detect/yolov9_homeobjects_150epochs/weights/best.pt",  # HPC trained
        "models/best_homeobjects.pt",  # Local copy
        "yolov9c.pt"  # Fallback
    ]
    
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                models['yolo'] = YOLO(path)
                st.sidebar.success(f"✅ Loaded: {path}")
                loaded = True
                break
            except Exception as e:
                continue
    
    if not loaded:
        models['yolo'] = YOLO("yolov9c.pt")
        st.sidebar.info("ℹ️ Using default YOLO model")
    
    # SAM for segmentation (optional)
    try:
        models['sam'] = SAM("sam_b.pt")
        st.sidebar.success("✅ Loaded SAM model")
    except:
        models['sam'] = None
        st.sidebar.warning("⚠️ SAM not available - using YOLO only")
    
    return models

models = load_models()

# -------------------------------
# Sidebar Settings
# -------------------------------
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    use_sam = st.checkbox("Use SAM for segmentation (slower)", value=False)
    
    st.header("🏠 Room Geometry")
    room_height_cm = st.number_input("Room height (cm)", 200, 350, 250)
    
    st.header("🌫️ Dust Simulation")
    fan_speed = st.slider("Fan Speed (%)", 0, 100, 50)
    window_open = st.checkbox("Window Open", value=False)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    aqi = st.slider("AQI (Air Quality Index)", 0, 500, 100)
    
    st.header("🎨 3D Visualization")
    show_pointcloud = st.checkbox("Show Point Cloud", value=True)
    show_trajectory = st.checkbox("Show Camera Trajectory", value=True)
    show_objects = st.checkbox("Show Object Bounding Boxes", value=True)
    show_particles = st.checkbox("Show Dust Particles", value=True)

# -------------------------------
# Helper Functions
# -------------------------------
def aggregate_2d_detections(all_detections):
    """Aggregate 2D detections by class."""
    class_groups = {}
    for det in all_detections:
        cls = det['class_name']
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(det)
    
    aggregated = []
    for cls, dets in class_groups.items():
        confs = [d['confidence'] for d in dets]
        
        # Average bbox size
        widths = [d['bbox'][2] - d['bbox'][0] for d in dets]
        heights = [d['bbox'][3] - d['bbox'][1] for d in dets]
        
        aggregated.append({
            'class_name': cls,
            'confidence': float(np.mean(confs)),
            'detections_count': len(dets),
            'width': float(np.mean(widths)) / 100.0,  # Convert to meters approx
            'height': float(np.mean(heights)) / 100.0,
            'depth': 0.5,  # Placeholder
            'center': [0, 0, 0],  # Placeholder for 3D
            'avg_bbox': (
                int(np.mean([d['bbox'][0] for d in dets])),
                int(np.mean([d['bbox'][1] for d in dets])),
                int(np.mean([d['bbox'][2] for d in dets])),
                int(np.mean([d['bbox'][3] for d in dets]))
            )
        })
    
    return aggregated

def create_3d_visualization(slam_results, objects_3d, dust_data=None):
    """Create 3D visualization."""
    fig = go.Figure()
    
    # 1. Point Cloud
    if show_pointcloud and slam_results.get('point_cloud') and len(slam_results['point_cloud']) > 0:
        pts = np.array(slam_results['point_cloud'])
        if len(pts) > 5000:
            idx = np.random.choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=1, color='lightgray', opacity=0.5),
            name='Point Cloud'
        ))
    
    # 2. Camera Trajectory
    if show_trajectory and slam_results.get('camera_trajectory'):
        poses = np.array([p[:3, 3] for p in slam_results['camera_trajectory']])
        fig.add_trace(go.Scatter3d(
            x=poses[:, 0], y=poses[:, 1], z=poses[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=3, color='red'),
            name='Camera Path'
        ))
    
    # 3. Object positions (2D projected)
    if show_objects and objects_3d:
        colors = {
            'bed': '#1f77b4', 'sofa': '#ff7f0e', 'chair': '#2ca02c',
            'table': '#d62728', 'lamp': '#9467bd', 'tv': '#8c564b',
            'laptop': '#e377c2', 'wardrobe': '#7f7f7f',
            'window': '#17becf', 'door': '#bcbd22',
            'potted plant': '#98df8a', 'photo frame': '#ff9896'
        }
        
        for i, obj in enumerate(objects_3d):
            color = colors.get(obj['class_name'], '#aaaaaa')
            
            # Place objects in 3D space (simple layout)
            angle = (i / max(len(objects_3d), 1)) * 2 * np.pi
            radius = 2.0
            x = radius * np.cos(angle)
            y = 0
            z = radius * np.sin(angle)
            
            # Add object marker
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=15, color=color, symbol='square'),
                text=[f"{obj['class_name']}<br>{obj['confidence']:.0%}"],
                textposition='top center',
                name=obj['class_name'],
                showlegend=False,
                hovertemplate=f"<b>{obj['class_name']}</b><br>" +
                             f"Confidence: {obj['confidence']:.1%}<br>" +
                             f"Detections: {obj['detections_count']}<extra></extra>"
            ))
    
    # 4. Dust Particles
    if dust_data is not None and show_particles:
        risk_map = dust_data.get('risk_map')
        if risk_map is not None:
            high_risk = np.where(risk_map == 3)
            if len(high_risk[0]) > 0:
                n_samples = min(500, len(high_risk[0]))
                indices = np.random.choice(len(high_risk[0]), n_samples, replace=False)
                particle_positions = np.array([
                    high_risk[0][indices],
                    high_risk[1][indices],
                    high_risk[2][indices]
                ]).T
                
                fig.add_trace(go.Scatter3d(
                    x=particle_positions[:, 0] * 0.1,
                    y=particle_positions[:, 1] * 0.1,
                    z=particle_positions[:, 2] * 0.1,
                    mode='markers',
                    marker=dict(size=5, color='red', opacity=0.5),
                    name='Dust Particles'
                ))
    
    fig.update_layout(
        title="3D Room Reconstruction with SLAM + Object Detection",
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgba(0,0,0,0)'
        ),
        width=900,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
        hovermode='closest'
    )
    
    return fig

def process_video_with_slam(uploaded_video, model, conf_thresh, use_sam=False):
    """Process video with SLAM + object detection pipeline (FIXED)."""
    # Save uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    try:
        # Step 1: Run SLAM
        st.write("🔄 Running SLAM on video...")
        slam = SLAMProcessor()
        slam_results = slam.process_video(video_path, frame_interval=10)
        
        # Step 2: Run object detection on keyframes
        st.write("🔍 Detecting objects in keyframes...")
        all_detections = []  # Store ALL detections (2D)
        progress_bar = st.progress(0)
        
        detection_frames = slam_results['detection_frames']
        
        for i, frame in enumerate(detection_frames):
            # Run YOLO detection
            results = model(frame, conf=conf_thresh, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                all_detections.append({
                    'class_name': model.names[int(box.cls[0])],
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(box.conf[0]),
                    'frame_idx': i
                })
            
            progress_bar.progress((i + 1) / len(detection_frames))
        
        # Step 3: Aggregate detections by class
        st.write(f"📊 Aggregating {len(all_detections)} detections...")
        aggregated = aggregate_2d_detections(all_detections)
        
        st.success(f"✅ Found {len(aggregated)} unique objects from {len(all_detections)} detections")
        
        # Step 4: Run dust simulation
        st.write("🌫️ Running dust simulation...")
        sim = DustSimulator(400, room_height_cm, 400)
        dust = sim.simulate_dust(fan_speed, window_open, humidity, aqi)
        risk = sim.classify_risk(dust)
        
        dust_data = {
            'concentration': dust,
            'risk_map': risk
        }
        
        return {
            'objects_3d': aggregated,
            'point_cloud': slam_results.get('point_cloud', []),
            'camera_trajectory': slam_results.get('camera_poses', []),
            'scale': slam_results.get('scale', 1.0),
            'keyframes_count': len(detection_frames),
            'dust_data': dust_data,
            'furniture': [],
            'all_detections': all_detections,
            'model_classes': list(model.names.values())
        }
        
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

# -------------------------------
# Main App
# -------------------------------
input_type = st.radio("Select input type:", ["📸 Image", "🎥 Video (Recommended)"], horizontal=True)

if input_type == "📸 Image":
    uploaded_img = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        st.info("Image processing with 3D reconstruction is simplified. Use Video for full 3D.")

else:  # Video
    uploaded_video = st.file_uploader(
        "Upload a room video (walk slowly, cover all corners)", 
        type=["mp4", "mov", "avi"],
        help="For best results: walk slowly, keep camera stable, cover all areas of the room"
    )
    
    if uploaded_video:
        with st.spinner("Processing video with SLAM and object detection..."):
            results = process_video_with_slam(
                uploaded_video, 
                models['yolo'], 
                conf_thresh,
                use_sam
            )
        
        # Display summary
        st.success(f"✅ Processed {results['keyframes_count']} keyframes, found {len(results['objects_3d'])} unique objects")
        
        # Show scale
        st.info(f"📏 Estimated scale: {results['scale']:.3f} cm/unit")
        
        # Show 3D objects table
        if results['objects_3d']:
            st.subheader("📊 3D Objects Detected")
            df = pd.DataFrame([{
                'Object': obj['class_name'].capitalize(),
                'Confidence': f"{obj['confidence']:.0%}",
                'Detections': obj['detections_count'],
                'Width (px)': f"{obj['width']*100:.0f}",
                'Height (px)': f"{obj['height']*100:.0f}"
            } for obj in results['objects_3d']])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
            if 'model_classes' in results:
                st.info(f"Model classes: {results['model_classes']}")
        
        # Risk zones
        dust_data = results.get('dust_data')
        if dust_data:
            risk = dust_data['risk_map']
            low_pct = np.sum(risk==1)/risk.size*100
            med_pct = np.sum(risk==2)/risk.size*100
            high_pct = np.sum(risk==3)/risk.size*100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🟢 Low Risk", f"{low_pct:.1f}%")
            col2.metric("🟡 Medium Risk", f"{med_pct:.1f}%")
            col3.metric("🔴 High Risk", f"{high_pct:.1f}%")
            col4.metric("Objects Found", len(results['objects_3d']))
        
        # 3D Visualization
        st.subheader("🏠 3D Room Reconstruction")
        
        fig = create_3d_visualization(
            results,
            results['objects_3d'],
            results.get('dust_data')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if dust_data:
            high_pct = np.sum(dust_data['risk_map']==3)/dust_data['risk_map'].size*100
            if high_pct > 30:
                st.error("⚠️ High dust risk detected! Increase ventilation.")
            elif high_pct > 10:
                st.warning("⚠️ Moderate dust risk. Consider opening windows.")
            else:
                st.success("✅ Low dust risk. Keep up with regular cleaning.")
        
        if fan_speed < 30:
            st.write("💨 Increase fan speed to improve air circulation.")
        if not window_open and humidity > 60:
            st.write("🪟 Open windows to reduce humidity.")
        if aqi > 150:
            st.write("🌫️ Poor outdoor air quality. Keep windows closed.")
        
        # Export data
        if st.button("📥 Export 3D Data (JSON)"):
            export_data = {
                'objects': results['objects_3d'],
                'furniture': results['furniture'],
                'scale': results['scale'],
                'keyframes': results['keyframes_count']
            }
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="aeroGuard_3d_data.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>AeroGuard – 3D Digital Twin for Indoor Dust Allergy Risk</p>
    <p>Powered by YOLO, SLAM, and 3D Visualization</p>
</div>
""", unsafe_allow_html=True)