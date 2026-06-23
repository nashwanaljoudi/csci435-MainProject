# Task 1 - Image Enhancement using CLAHE
# Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
# luminance channel of the input frame to improve contrast before segmentation.

import cv2  # OpenCV library for all image processing operations


def enhance(frame):
    """
    Enhance a raw BGR webcam frame so the hand is clearly visible
    even under poor or uneven lighting conditions.

    Strategy: convert to LAB colour space, apply CLAHE only to the
    lightness channel (L), then convert back to BGR.  Touching only L
    preserves the original skin/background colours while fixing contrast.

    Args:
        frame: numpy array of shape (H, W, 3), dtype uint8, BGR channel order
               — exactly what cv2.VideoCapture.read() returns.

    Returns:
        enhanced: numpy array with the same shape and dtype as frame,
                  but with improved local contrast.
    """

    # --- Step 1: Convert BGR → LAB ---
    # OpenCV reads frames in Blue-Green-Red (BGR) order by convention.
    # LAB is a perceptual colour space with three channels:
    #   L  – Lightness (0 = black, 255 = white); this is what CLAHE will fix.
    #   A  – green-to-red colour opponent axis.
    #   B  – blue-to-yellow colour opponent axis.
    # We work in LAB because L is completely separated from colour information,
    # so equalising it won't shift hues — only brightness/contrast changes.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # --- Step 2: Split LAB into its three individual channels ---
    # cv2.split() returns a tuple of single-channel arrays.
    # We name them l, a, b to match the LAB notation.
    # Only 'l' will be modified; 'a' and 'b' are kept untouched.
    l, a, b = cv2.split(lab)

    # --- Step 3: Create a CLAHE object ---
    # CLAHE = Contrast Limited Adaptive Histogram Equalization.
    # Unlike plain histogram equalisation (which is global), CLAHE divides the
    # image into small rectangular tiles and equalises each tile independently.
    # This fixes local contrast problems — e.g. a bright window behind the hand
    # that would otherwise wash out the entire image.
    #
    # clipLimit=2.0  → maximum slope of the CDF inside each tile.
    #   A higher value allows more contrast amplification but also amplifies
    #   noise.  2.0 is a widely-used default that balances enhancement vs noise.
    #
    # tileGridSize=(8, 8) → the image is split into an 8×8 grid of tiles.
    #   Smaller tiles react to finer local variations; larger tiles behave more
    #   like global equalisation.  8×8 works well for typical webcam resolutions
    #   (480p – 1080p) and hand-detection tasks.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # --- Step 4: Apply CLAHE to the L channel only ---
    # clahe.apply() takes a single-channel uint8 array and returns one.
    # The result 'l_enhanced' has the same spatial dimensions as 'l' but with
    # locally equalised contrast — the hand's edges and creases become crisper.
    l_enhanced = clahe.apply(l)

    # --- Step 5: Merge the enhanced L back with the unchanged A and B ---
    # cv2.merge() recombines the three single-channel arrays into one
    # three-channel array.  The colour channels 'a' and 'b' are unchanged, so
    # skin tone is preserved — only brightness distribution is improved.
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # --- Step 6: Convert LAB → BGR ---
    # We convert back to BGR so the output frame is in the same format that
    # every downstream step (segmentation, display, etc.) expects.
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Return the contrast-enhanced frame, ready for Task 2 (segmentation).
    return enhanced
