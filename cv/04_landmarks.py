# Task 4 - MediaPipe Hands landmark extraction (pre-trained model)
# Uses Google MediaPipe Hands to detect 21 hand keypoints (x, y, z) from the
# cropped hand region. These landmarks are used as features for classification.

import cv2
import mediapipe as mp

_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
_draw = mp.solutions.drawing_utils
_style = mp.solutions.drawing_styles


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

    # Crop the frame to the bounding box so MediaPipe works on the hand only
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]

    if crop.size == 0:
        return None, annotated

    # Run MediaPipe Hands on the cropped region (expects RGB input)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None, annotated

    hand_landmarks = results.multi_hand_landmarks[0]

    # Draw the MediaPipe skeleton and red landmark dots onto the crop
    _draw.draw_landmarks(crop, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                         _style.get_default_hand_landmarks_style(),
                         _style.get_default_hand_connections_style())
    for lm in hand_landmarks.landmark:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(crop, (cx, cy), 4, (0, 0, 255), -1)

    annotated[y:y + h, x:x + w] = crop

    # Extract raw (x, y, z) for all 21 landmarks
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Normalise each coordinate relative to the wrist (landmark 0)
    wx, wy, wz = points[0]
    features = []
    for px, py, pz in points:
        features.extend([px - wx, py - wy, pz - wz])

    return features, annotated
