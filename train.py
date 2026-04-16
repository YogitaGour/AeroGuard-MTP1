# train.py
from ultralytics import YOLO
import os
import yaml

def ensure_model():
    if not os.path.exists('yolov8n.pt'):
        print("Downloading yolov8n.pt...")
        YOLO('yolov8n.pt')
        print("Download complete.")

def check_dataset(data_yaml):
    """Check if dataset folders exist."""
    try:
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading YAML: {e}")
        return False
    
    if config is None:
        print(f"❌ YAML file {data_yaml} is empty or invalid.")
        return False
    
    # Print what we loaded
    print("YAML content:", config)
    
    base = config.get('path', '')
    train_rel = config.get('train', '')
    val_rel = config.get('val', '')
    
    if not base or not train_rel or not val_rel:
        print("❌ Missing 'path', 'train', or 'val' in YAML.")
        return False
    
    train_dir = os.path.join(base, train_rel)
    val_dir = os.path.join(base, val_rel)
    
    print(f"Looking for train folder: {train_dir}")
    print(f"Looking for val folder: {val_dir}")
    
    if not os.path.exists(train_dir):
        print(f"❌ Train folder missing: {train_dir}")
        return False
    if not os.path.exists(val_dir):
        print(f"❌ Validation folder missing: {val_dir}")
        return False
    
    print("✅ Dataset found!")
    return True

def main():
    data_yaml = "config/homeobjects.yaml"
    
    ensure_model()
    
    if not check_dataset(data_yaml):
        print("Aborting training.")
        return
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train
    results=model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        name='homeobjects_finetuned',
        device='cpu',
        workers=0,
        patience=5,
        save=True,
        save_period=5,
        pretrained=True,
        optimizer='auto'
    )
    
     # Get the actual save directory from the trainer
    save_dir = results.save_dir   # or results.trainer.save_dir
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    
    print(f"Looking for best model at: {best_model_path}")
    
    if os.path.exists(best_model_path):
        os.makedirs('models', exist_ok=True)
        import shutil
        shutil.copy(best_model_path, 'models/best.pt')
        print(f"✅ Model saved to models/best.pt")
    else:
        print("❌ Training failed – best model not found")



if __name__ == "__main__":
    main()