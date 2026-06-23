# Task 4 - MediaPipe Hands landmark extraction (pre-trained model)
# Uses Google MediaPipe Hand Landmarker to detect 21 hand keypoints (x, y, z)
# from the cropped hand region. These landmarks are used as features for classification.

import cv2

from cv.hand_utils import VideoHandLandmarker, draw_hand_landmarks, landmarks_to_features

_hands = VideoHandLandmarker()


def get_landmarks(frame, bbox):
    """
    Extract and normalise 21 MediaPipe hand landmarks from a cropped region.

    Args:
        frame: numpy array (H, W, 3), dtype uint8, BGR — output of enhance().
        bbox:  (x, y, w, h) bounding box returned by get_hand_region(), or None.

    Returns:
        features:  list of 63 floats [x0,y0,z0, x1,y1,z1, …] normalised to wrist,
                   or None if no hand is detected.
        annotated: copy of frame with MediaPipe skeleton and red landmark dots drawn.
    """

    annotated = frame.copy()

    if bbox is None:
        return None, annotated

    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]

    if crop.size == 0:
        return None, annotated

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = _hands.detect(rgb)

    if not results.hand_landmarks:
        return None, annotated

    hand_landmarks = results.hand_landmarks[0]
    crop = draw_hand_landmarks(crop.copy(), hand_landmarks)
    annotated[y:y + h, x:x + w] = crop

    return landmarks_to_features(hand_landmarks), annotated
