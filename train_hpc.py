# train_hpc.py - YOLOv9 Training on IITJ HPC (Auto GPU Detection)

import os
import subprocess
import torch
from ultralytics import YOLO

print("=" * 60)
print("🚀 YOLOv9 TRAINING ON HPC")
print("=" * 60)

# Auto-detect MIG UUID on current node
def get_mig_uuid():
    """Automatically find available MIG UUID on current node"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "MIG" in line and "UUID" in line:
                # Extract UUID from line like: MIG 1g.6gb Device 0: (UUID: MIG-...)
                uuid = line.split("UUID:")[1].strip().rstrip(")")
                return uuid
    except Exception as e:
        print(f"⚠️ MIG detection failed: {e}")
    return None

# Set MIG UUID if available
mig_uuid = get_mig_uuid()
if mig_uuid:
    os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuid
    print(f"✅ MIG UUID detected: {mig_uuid}")
else:
    print("ℹ️ No MIG UUID found — using default GPU")

# Verify GPU
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("\n❌ GPU not available! Exiting.")
    exit(1)

# Load YOLOv9c
print("\n📥 Loading YOLOv9c...")
model = YOLO("yolov9c.pt")

# Train
print("\n🔥 Starting training...")
results = model.train(
    data="HomeObjects-3K.yaml",
    epochs=150,                    # Test ke liye 1 epoch
    batch=8,
    imgsz=640,
    device=0,
    workers=4,
    save=True,
    save_period=1,
    name="yolov9_homeobjects_150epochs",
    plots=True,
    augment=True,
    lr0=0.01,
)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"📁 Results saved to: {results.save_dir}")
