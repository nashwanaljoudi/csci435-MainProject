# Shared MediaPipe Hand Landmarker helpers (Tasks Vision API).
# Compatible with mediapipe >= 0.10.30 on Python 3.10+.

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "hand_landmarker.task"


def ensure_model_path() -> str:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def landmarks_to_features(landmarks) -> list[float]:
    points = [(lm.x, lm.y, lm.z) for lm in landmarks]
    wx, wy, wz = points[0]
    features = []
    for px, py, pz in points:
        features.extend([px - wx, py - wy, pz - wz])
    return features


def draw_hand_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
    return image


class VideoHandLandmarker:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model_path()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, rgb_image):
        self._timestamp_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

    def close(self):
        self._landmarker.close()


class ImageHandLandmarker:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model_path()),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self._landmarker.detect(mp_image)

    def close(self):
        self._landmarker.close()
