# train_homeobjects.py
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")  # start from a pretrained model

# Train the model using the dataset configuration
results = model.train(
    data="/Users/nomeshgaur/Desktop/ AeroGuard/HomeObjects-3K.yaml",  # path to the dataset YAML
    epochs=100,          # number of training epochs
    imgsz=640,          # input image size
    batch=16,           # batch size (reduce if you run out of memory)
    name="homeobjects_finetuned",  # name of the training run
    device="cpu",       # use "cuda" if you have a GPU
    workers=0,          # set to 0 on macOS to avoid multiprocessing issues
    patience=10,        # early stopping patience
    save=True,          # save checkpoints
    pretrained=True,    # use pretrained weights
)

# After training, copy best model to models/
import shutil, os
best = "runs/detect/homeobjects_finetuned/weights/best.pt"
if os.path.exists(best):
    os.makedirs("models", exist_ok=True)
    shutil.copy(best, "models/best.pt")
    print("✅ Model saved to models/best.pt")