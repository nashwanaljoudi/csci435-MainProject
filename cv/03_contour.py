# Task 3 - Hand Contour Detection and Bounding Box
# Finds contours in the segmentation mask, selects the largest one (the hand),
# draws the contour outline, and computes a tight bounding box for cropping.

import cv2


def get_hand_region(mask, frame):
    """
    Detect the hand contour in a binary mask and annotate the source frame.

    Args:
        mask:  numpy array (H, W), dtype uint8 — output of segment().
        frame: numpy array (H, W, 3), dtype uint8, BGR — output of enhance().

    Returns:
        (x, y, w, h): bounding box of the largest contour, or None if no hand found.
        annotated:    copy of frame with the hand contour drawn on it.
    """

    # Run Canny on the mask to get sharp contour edges
    edges = cv2.Canny(mask, threshold1=50, threshold2=150)

    # Find all external contours from the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame.copy()

    if not contours:
        return None, annotated

    # Pick the largest contour by area — that's the hand
    hand_contour = max(contours, key=cv2.contourArea)

    # Draw the contour outline on the frame copy
    cv2.drawContours(annotated, [hand_contour], -1, color=(0, 255, 0), thickness=2)

    # Compute the tight bounding box around the hand contour
    x, y, w, h = cv2.boundingRect(hand_contour)

    return (x, y, w, h), annotated
