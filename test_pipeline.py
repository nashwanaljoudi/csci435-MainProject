# This file is for testing the CV pipeline without the website.
# Run: python test_pipeline.py
# It will open a webcam window, run process_frame() on each frame,
# and display the annotated output directly using OpenCV — no Streamlit needed.

import time
import cv2
from cv.pipeline import process_frame

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

prev_time = time.monotonic()

# Track the last prediction so the overlay persists between frames
_last_letter = ""
_last_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the full 7-step pipeline
    annotated, committed, confidence, word, sentence = process_frame(frame)

    # Compute and overlay FPS in the top-left corner
    now = time.monotonic()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Update cached prediction whenever the pipeline returns one
    if confidence is not None and confidence > 0:
        _last_letter = committed if committed else _last_letter
        _last_conf = confidence

    # Print to terminal only when a new letter is committed
    if committed is not None:
        print(f"✅ Letter: {committed} | Confidence: {int(_last_conf * 100)}% | Word: {word} | Sentence: {sentence}")

    # Draw the current predicted letter in large text at the bottom of the frame
    h, w = annotated.shape[:2]
    if _last_letter:
        label = f"{_last_letter}  {int(_last_conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 6)
        tx = (w - tw) // 2
        ty = h - 30
        # Dark background strip for readability
        cv2.rectangle(annotated, (0, ty - th - 20), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 0), 6)

    cv2.imshow("ASL Pipeline", annotated)

    # Quit on Q key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
