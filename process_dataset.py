# Extract MediaPipe hand landmarks from ASL alphabet images and save as a
# dataset pickle. Each subfolder under --dataset-dir is treated as a class label.
# Usage: python process_dataset.py --dataset-dir path/to/asl_alphabet

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=True, help="Root folder with one subfolder per letter")
parser.add_argument("--max-per-class", type=int, default=500, help="Max images to process per letter")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
max_per_class = args.max_per_class

# Set up MediaPipe Hands in static-image mode for offline processing
hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

X, y = [], []

# Iterate over each class subfolder (one letter per folder)
class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

for class_dir in class_dirs:
    label = class_dir.name
    image_paths = list(class_dir.glob("*"))[:max_per_class]

    for img_path in tqdm(image_paths, desc=label, unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Run MediaPipe on the RGB image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            continue

        # Extract and normalise 21 landmarks relative to the wrist (landmark 0)
        lms = results.multi_hand_landmarks[0].landmark
        points = [(lm.x, lm.y, lm.z) for lm in lms]
        wx, wy, wz = points[0]
        features = []
        for px, py, pz in points:
            features.extend([px - wx, py - wy, pz - wz])

        X.append(features)
        y.append(label)

hands.close()

# Save the landmark dataset to disk
Path("data").mkdir(exist_ok=True)
with open("data/dataset.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print(f"\nSaved {len(X)} samples across {len(class_dirs)} classes to data/dataset.pkl")
