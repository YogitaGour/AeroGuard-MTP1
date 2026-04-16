from pathlib import Path

dataset_root = Path("datasets/HomeObjects-3K")
label_files = list(dataset_root.rglob("*.txt"))

if not label_files:
    print("No label files found. Check the dataset folder structure.")
else:
    class_ids = set()
    for label_file in label_files:
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    print("Classes found:", sorted(class_ids))