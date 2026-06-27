# Task 1 - Image Enhancement using CLAHE
# Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
# luminance channel of the input frame to improve contrast before segmentation.

import cv2


def enhance(frame):
    # Convert to LAB so we can touch only the lightness channel (L),
    # leaving colours unchanged
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE improves local contrast — better than global equalisation
    # because it handles uneven lighting (e.g. bright window behind the hand)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge enhanced L back with original colour channels and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
