# Extract MediaPipe hand landmarks from ASL alphabet images and save as a
# dataset pickle. Each subfolder under --dataset-dir is treated as a class label.
# Usage: python process_dataset.py --dataset-dir path/to/asl_alphabet

import argparse
import pickle
from pathlib import Path

import cv2
from tqdm import tqdm

from cv.hand_utils import ImageHandLandmarker, landmarks_to_features

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=True, help="Root folder with one subfolder per letter")
parser.add_argument("--max-per-class", type=int, default=500, help="Max images to process per letter")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
max_per_class = args.max_per_class

hands = ImageHandLandmarker()
X, y = [], []

class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

for class_dir in class_dirs:
    label = class_dir.name
    image_paths = list(class_dir.glob("*"))[:max_per_class]

    for img_path in tqdm(image_paths, desc=label, unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.detect(rgb)

        if not results.hand_landmarks:
            continue

        X.append(landmarks_to_features(results.hand_landmarks[0]))
        y.append(label)

hands.close()

Path("data").mkdir(exist_ok=True)
with open("data/dataset.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print(f"\nSaved {len(X)} samples across {len(class_dirs)} classes to data/dataset.pkl")
