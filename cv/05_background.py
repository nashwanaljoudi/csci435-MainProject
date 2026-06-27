# Task 5 - Background Modelling using MOG2
# Uses OpenCV's MOG2 background subtractor to isolate moving foreground objects
# (the hand) from a static or slowly changing background in video streams.

import cv2

# Persistent subtractor — shared across all calls so the background model accumulates
_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


def hand_present(frame):
    # Subtract learned background to get a foreground mask
    fg_mask = _mog2.apply(frame)

    # Remove noise and small blobs with opening, then expand remaining regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # Treat frame as hand-present only if enough foreground pixels exist
    return cv2.countNonZero(fg_mask) > 500
