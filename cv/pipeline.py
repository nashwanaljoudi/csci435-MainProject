# Ties all 7 tasks together into a single process_frame(frame) function.
# Input:  a raw BGR frame (numpy array) from webcam or video file
# Output: annotated frame + dict with keys: letter, confidence, word_so_far
# Call order: enhance (1) → segment (2) → contour (3) → landmarks (4) → background (5) → classify (6) → smooth (7)
