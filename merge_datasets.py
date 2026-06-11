# Extracts MediaPipe landmarks from two ASL datasets and merges them into
# one combined dataset.pkl for training.
# Usage: python merge_datasets.py

import pickle
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

DATASETS = [
    {
        "path": Path("/Users/marinaelkedwany/Downloads/ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/asl_dataset"),
        "max_per_class": 1000,
        "name": "ASL_HG",
    },
    {
        "path": Path("/Users/marinaelkedwany/Downloads/archive-2/asl_alphabet_train/asl_alphabet_train"),
        "max_per_class": 3000,
        "name": "Kaggle",
    },
]

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

X, y = [], []

for dataset in DATASETS:
    print(f"\n--- Processing {dataset['name']} ({dataset['path'].name}) ---")
    class_dirs = sorted([d for d in dataset["path"].iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        label = class_dir.name.upper()
        image_paths = [p for p in class_dir.glob("*") if not p.name.startswith(".")][:dataset["max_per_class"]]

        for img_path in tqdm(image_paths, desc=f"{dataset['name']}/{label}", unit="img"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                continue

            lms = results.multi_hand_landmarks[0].landmark
            points = [(lm.x, lm.y, lm.z) for lm in lms]
            wx, wy, wz = points[0]
            features = []
            for px, py, pz in points:
                features.extend([px - wx, py - wy, pz - wz])

            X.append(features)
            y.append(label)

hands.close()

# Save merged dataset
Path("data").mkdir(exist_ok=True)
with open("data/dataset.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

from collections import Counter
counts = Counter(y)
print(f"\nMerged dataset saved — {len(X)} total samples across {len(counts)} classes")
for label in sorted(counts):
    print(f"  {label:8s}: {counts[label]}")
