# create_synthetic_data.py
import os
import random
import cv2
import numpy as np

def create_synthetic_dataset(output_dir='datasets/HomeObjects', num_train=100, num_val=20):
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)

    class_names = ['bed', 'sofa', 'chair', 'table', 'lamp', 'tv', 'laptop', 'wardrobe', 'window', 'door', 'potted plant', 'picture frame']
    num_classes = len(class_names)

    for split, num in [('train', num_train), ('val', num_val)]:
        for i in range(num):
            img = np.ones((640, 640, 3), dtype=np.uint8) * 255
            cv2.imwrite(f'{output_dir}/images/{split}/img_{i:04d}.jpg', img)

            with open(f'{output_dir}/labels/{split}/img_{i:04d}.txt', 'w') as f:
                # Generate 2-5 random objects per image
                for _ in range(random.randint(2, 5)):
                    cls = random.randint(0, num_classes-1)
                    x = random.uniform(0.1, 0.9)
                    y = random.uniform(0.1, 0.9)
                    w = random.uniform(0.05, 0.3)
                    h = random.uniform(0.05, 0.3)
                    f.write(f"{cls} {x:.4f} {y:.4f} {w:.4f} {h:.4f}\n")

    print(f"✅ Synthetic dataset created in {output_dir}")

if __name__ == '__main__':
    create_synthetic_dataset()