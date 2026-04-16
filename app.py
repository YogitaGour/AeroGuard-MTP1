# app.py – Full AeroGuard with All Fixes
import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from src.temporal_tracker import TemporalTracker
from src.dust_simulator import DustSimulator
from src.visualizer import plot_risk_heatmap
import os
import random
import plotly.graph_objects as go

st.set_page_config(page_title="AeroGuard", layout="wide")
st.title("🛡️ AeroGuard – A 3D Digital Twin Simulator for Indoor Dust Allergy Risk Assessment")

# -------------------------------
# Load fine-tuned model
# -------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("models/best_homeobjects.pt"):
        model = YOLO("models/best_homeobjects.pt")
        st.sidebar.success("✅ Loaded fine-tuned model (HomeObjects-3K)")
    else:
        model = YOLO("yolov8n.pt")
        st.sidebar.info("ℹ️ Using default YOLOv8n model")
    return model

model = load_model()

# -------------------------------
# Sidebar settings
# -------------------------------
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.05,
                            help="Lower = more detections, but may include false positives. We ignore <10%.")
    show_all_objects = st.checkbox("Show all detected objects in table", value=False,
                                   help="If unchecked, only furniture (bed, sofa, chair, table, lamp, TV, fridge, wardrobe) are shown.")
    manual_scale = st.number_input("Manual scale (cm/pixel) – 0 to use auto", value=0.0, step=0.05,
                                   help="Override automatic calibration. Typical values: 0.2–0.5.")
    
    st.header("🌫️ Dust Simulation Parameters")
    fan_speed = st.slider("Fan Speed (%)", 0, 100, 50)
    window_open = st.checkbox("Window Open", value=False)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    aqi = st.slider("AQI (Air Quality Index)", 0, 500, 100)
    
    st.header("🏠 Room Geometry (approximate)")
    room_height_cm = st.number_input("Room height (cm)", 200, 350, 250)
    room_width_cm = st.number_input("Room width (cm)", 200, 800, 400)
    room_depth_cm = st.number_input("Room depth (cm)", 200, 800, 400)
    
    st.markdown("---")
    st.info("Dust risk is based on furniture layout and environmental factors, not on confidence threshold.")

# -------------------------------
# Helper functions
# -------------------------------
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def nms(detections, iou_threshold=0.5, min_conf=0.25):
    """Better NMS to remove duplicate detections"""
    
    # Filter low confidence first
    detections = [d for d in detections if d['confidence'] >= min_conf]
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    final_detections = []

    for det in detections:
        keep = True
        
        for kept in final_detections:
            iou = compute_iou(det['bbox'], kept['bbox'])
            
            # 🔥 Remove if overlapping too much (duplicate)
            if iou > iou_threshold:
                keep = False
                break
        
        if keep:
            final_detections.append(det)

    return final_detections

def calibrate_scale(detections, known_object='door', known_height_cm=200):
    for d in detections:
        if d['class_name'].lower() == known_object and d['height_px'] > 0:
            scale = known_height_cm / d['height_px']
            return scale, known_object
    return None, None

def detections_to_dataframe(detections, scale):
    """Convert detections to DataFrame (all objects)."""
    rows = []
    for d in detections:
        width_cm = d['width_px'] * scale
        height_cm = d['height_px'] * scale
        rows.append({
            "Object": d['class_name'].capitalize(),
            "Width (cm)": round(width_cm, 1),
            "Height (cm)": round(height_cm, 1),
            "Confidence": f"{d['confidence']:.0%}"
        })
    return pd.DataFrame(rows)

def detections_to_furniture(detections, scale, room_width, room_depth):
    """Returns (furniture_list, df_all, df_furniture)."""
    furniture = []
    all_rows = []
    furniture_rows = []
    furniture_classes = ['bed','sofa','chair','table','lamp','tv','refrigerator','wardrobe']
    
    for d in detections:
        width_cm = d['width_px'] * scale
        height_cm = d['height_px'] * scale
        depth_cm = width_cm * 0.8   # rough assumption
        
        # Estimate position
        if 'bbox' in d:
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            cy = (d['bbox'][1] + d['bbox'][3]) / 2
            x = cx / 640 * room_width   # assume YOLO input size 640
            z = cy / 640 * room_depth
        else:
            x = random.uniform(width_cm/2, room_width - width_cm/2)
            z = random.uniform(depth_cm/2, room_depth - depth_cm/2)
        x = max(0, min(room_width - width_cm, x))
        z = max(0, min(room_depth - depth_cm, z))
        y = 0  # floor
        
        row = {
            "Object": d['class_name'].capitalize(),
            "Width (cm)": round(width_cm, 1),
            "Height (cm)": round(height_cm, 1),
            "Confidence": f"{d['confidence']:.0%}"
        }
        all_rows.append(row)
        if d['class_name'].lower() in furniture_classes:
            furniture_rows.append(row)
            furniture.append((x, y, z, width_cm, height_cm, depth_cm))
    
    df_all = pd.DataFrame(all_rows)
    df_furniture = pd.DataFrame(furniture_rows)
    return furniture, df_all, df_furniture

def create_3d_room_visualization(furniture, room_width, room_height, room_depth):
    """Create a 3D Plotly figure with room wireframe and furniture cubes."""
    fig = go.Figure()
    # Room wireframe (floor)
    floor_corners = [[0,0,0], [room_width,0,0], [room_width,0,room_depth], [0,0,room_depth]]
    for i in range(4):
        fig.add_trace(go.Scatter3d(
            x=[floor_corners[i][0], floor_corners[(i+1)%4][0]],
            y=[floor_corners[i][1], floor_corners[(i+1)%4][1]],
            z=[floor_corners[i][2], floor_corners[(i+1)%4][2]],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))
    # Vertical lines
    for x,y,z in floor_corners[:2]:
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[0, room_height], z=[z, z],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))
    # Furniture as cubes
    for (x,y,z,w,h,d) in furniture:
        # 8 vertices of the cube
        vertices = np.array([
            [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
            [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d]
        ])
        faces = np.array([
            [0,1,2,3], [4,5,6,7], [0,1,5,4],
            [3,2,6,7], [0,3,7,4], [1,2,6,5]
        ])
        for face in faces:
            fig.add_trace(go.Mesh3d(
                x=vertices[face,0], y=vertices[face,1], z=vertices[face,2],
                color='lightblue', opacity=0.6, showlegend=False
            ))
    fig.update_layout(
        title="Simplified 3D Room Layout",
        scene=dict(
            xaxis_title="Width (cm)", yaxis_title="Height (cm)", zaxis_title="Depth (cm)",
            aspectmode='data'
        ),
        width=700, height=500
    )
    return fig

# -------------------------------
# Input type selection
# -------------------------------
input_type = st.radio("Select input type:", ["📸 Image", "🎥 Video"], horizontal=True)

MIN_OBJECTS = 2
DEFAULT_FURNITURE = [
    (100, 0, 100, 90, 190, 90),   # bed
    (300, 0, 200, 60, 75, 60),    # table
    (250, 0, 250, 50, 80, 50),    # chair
]

# -------------------------------
# IMAGE PROCESSING
# -------------------------------
if input_type == "📸 Image":
    uploaded_img = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Run detection
        results = model(img_np, conf=conf_thresh)
        annotated = results[0].plot()
        with col2:
            st.image(annotated, caption="Detected Objects", use_container_width=True)
        
        # Extract raw detections
        raw_dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            raw_dets.append({
                'class_name': model.names[int(box.cls[0])],
                'bbox': (x1, y1, x2, y2),
                'width_px': x2 - x1,
                'height_px': y2 - y1,
                'confidence': float(box.conf[0])
            })
        # Apply NMS and confidence filter
        detections = nms(raw_dets, iou_threshold=0.3, min_conf=0.1)
        
        if detections:
            # Scale calibration
            if manual_scale > 0:
                scale = manual_scale
                st.info(f"Using manual scale: {scale:.3f} cm/pixel")
            else:
                scale, ref_obj = calibrate_scale(detections)
                if scale is None:
                    scale, ref_obj = calibrate_scale(detections, known_object='window', known_height_cm=120)
                if scale is None:
                    scale = 0.25
                    st.warning("No door or window detected. Using default scale (0.25 cm/pixel).")
                else:
                    st.success(f"Calibrated using {ref_obj}: scale = {scale:.3f} cm/pixel")
            
            # Convert to furniture and DataFrames
            furniture, df_all, df_furniture = detections_to_furniture(detections, scale, room_width_cm, room_depth_cm)
            st.subheader("📊 Detected Objects")
            if show_all_objects:
                st.dataframe(df_all, use_container_width=True)
            else:
                st.dataframe(df_furniture, use_container_width=True)
            
            # Dust simulation uses furniture (even if less than MIN_OBJECTS, we still use detected)
            if len(furniture) < MIN_OBJECTS:
                st.warning(f"Only {len(furniture)} furniture pieces detected. Using default furniture layout for simulation.")
                furniture = DEFAULT_FURNITURE
        else:
            st.warning("No objects detected after filtering. Using default furniture layout.")
            furniture = DEFAULT_FURNITURE
            df_all = df_furniture = pd.DataFrame()
        
        # Dust simulation (always runs)
        sim = DustSimulator(room_width_cm, room_height_cm, room_depth_cm)
        for (x, y, z, w, h, d) in furniture:
            sim.add_furniture(x, y, z, w, h, d)
        dust = sim.simulate_dust(fan_speed, window_open, humidity, aqi)
        risk = sim.classify_risk(dust)
        
        low_pct = np.sum(risk==1)/risk.size*100
        med_pct = np.sum(risk==2)/risk.size*100
        high_pct = np.sum(risk==3)/risk.size*100
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Low Risk Zones", f"{low_pct:.1f}%")
        col2.metric("🟡 Medium Risk Zones", f"{med_pct:.1f}%")
        col3.metric("🔴 High Risk Zones", f"{high_pct:.1f}%")
        
        # 3D heatmap
        fig_heat = plot_risk_heatmap(dust)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # 3D room visualisation
        fig_room = create_3d_room_visualization(furniture, room_width_cm, room_height_cm, room_depth_cm)
        st.plotly_chart(fig_room, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if high_pct > 30:
            st.error("⚠️ High dust risk detected! Increase ventilation, clean more often, or use an air purifier.")
        elif high_pct > 10:
            st.warning("⚠️ Moderate dust risk. Consider opening windows or reducing humidity.")
        else:
            st.success("✅ Low dust risk. Keep up with regular cleaning.")
        if fan_speed < 30:
            st.write("💨 Increase fan speed to improve air circulation.")
        if not window_open and humidity > 60:
            st.write("🪟 Open windows to reduce humidity and dust concentration.")
        if aqi > 150:
            st.write("🌫️ Poor outdoor air quality. Keep windows closed and use an air purifier.")
        print("Dust value in image",np.max(dust))
# -------------------------------
# VIDEO PROCESSING (similar but uses temporal tracker)
# -------------------------------
else:
    uploaded_video = st.file_uploader("Upload a room video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        tracker = TemporalTracker()
        all_detections = []
        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        status = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 5 == 0:
                results = model(frame, conf=conf_thresh)
                dets = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    dets.append({
                        'class_name': model.names[int(box.cls[0])],
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf[0])
                    })
                tracked = tracker.update(dets)  # returns list with width_px, height_px, confidence, class_name
                # Add width_px, height_px to tracked (tracker already has them)
                all_detections.extend(tracked)
            frame_count += 1
            progress.progress(frame_count / total_frames)
            status.text(f"Processing frame {frame_count}/{total_frames}")
        cap.release()
        status.text("Processing complete!")
        
        # Filter low confidence and apply NMS (tracker already does some merging)
        all_detections = [d for d in all_detections if d.get('confidence', 0) >= 0.1]
        
        if all_detections:
            if manual_scale > 0:
                scale = manual_scale
                st.info(f"Using manual scale: {scale:.3f} cm/pixel")
            else:
                scale, ref_obj = calibrate_scale(all_detections)
                if scale is None:
                    scale, ref_obj = calibrate_scale(all_detections, known_object='window', known_height_cm=120)
                if scale is None:
                    scale = 0.25
                    st.warning("No door or window detected. Using default scale (0.25 cm/pixel).")
                else:
                    st.success(f"Calibrated using {ref_obj}: scale = {scale:.3f} cm/pixel")
            
            furniture, df_all, df_furniture = detections_to_furniture(all_detections, scale, room_width_cm, room_depth_cm)
            st.subheader("📊 Detected Objects")
            if show_all_objects:
                st.dataframe(df_all, use_container_width=True)
            else:
                st.dataframe(df_furniture, use_container_width=True)
            
            if len(furniture) < MIN_OBJECTS:
                st.warning(f"Only {len(furniture)} furniture pieces detected. Using default furniture layout.")
                furniture = DEFAULT_FURNITURE
        else:
            st.warning("No objects detected after filtering. Using default furniture layout.")
            furniture = DEFAULT_FURNITURE
        
        # Dust simulation
        sim = DustSimulator(room_width_cm, room_height_cm, room_depth_cm)
        for (x, y, z, w, h, d) in furniture:
            sim.add_furniture(x, y, z, w, h, d)
        dust = sim.simulate_dust(fan_speed, window_open, humidity, aqi)
        risk = sim.classify_risk(dust)
        
        low_pct = np.sum(risk==1)/risk.size*100
        med_pct = np.sum(risk==2)/risk.size*100
        high_pct = np.sum(risk==3)/risk.size*100
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Low Risk Zones", f"{low_pct:.1f}%")
        col2.metric("🟡 Medium Risk Zones", f"{med_pct:.1f}%")
        col3.metric("🔴 High Risk Zones", f"{high_pct:.1f}%")
        
        fig_heat = plot_risk_heatmap(dust)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        fig_room = create_3d_room_visualization(furniture, room_width_cm, room_height_cm, room_depth_cm)
        st.plotly_chart(fig_room, use_container_width=True)
        
        st.subheader("💡 Recommendations")
        if high_pct > 30:
            st.error("⚠️ High dust risk detected! Improve ventilation and cleaning.")
        elif high_pct > 10:
            st.warning("⚠️ Moderate dust risk. Increase fan speed or open windows.")
        else:
            st.success("✅ Low dust risk. Good air quality!")  
         
        np.max(dust)
        print("Dust value in video",np.max(dust))

