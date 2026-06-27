# Task 2 - Skin Segmentation using HSV and morphological operations
# Converts the enhanced frame to HSV colour space, applies a skin-colour range
# mask, then cleans the mask with erosion/dilation to remove noise and fill gaps.

import cv2
import numpy as np


def segment(frame):
    """
    Segment skin pixels from an enhanced BGR frame and return a binary mask.

    Args:
        frame: numpy array (H, W, 3), dtype uint8, BGR — output of enhance().

    Returns:
        mask: numpy array (H, W), dtype uint8, 255 = hand pixel, 0 = background.
    """

    # Convert to HSV so skin colour can be isolated by hue alone
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold to keep only pixels whose hue/saturation/value fall in the skin range
    lower_skin = np.array([0, 15, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Erode to remove small noise blobs, then dilate to restore hand shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=4)

    return mask
