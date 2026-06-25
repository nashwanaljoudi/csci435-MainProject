# Ties all 7 tasks together into a single process_frame(frame) function.
# Input:  a raw BGR frame (numpy array) from webcam or video file
# Output: annotated frame + dict with keys: letter, confidence, word_so_far
# Call order: enhance (1) → segment (2) → contour (3) → landmarks (4) → background (5) → classify (6) → smooth (7)

import importlib.util
from pathlib import Path

# Load each task module by file path — names starting with digits aren't valid Python identifiers
_here = Path(__file__).parent

def _load(filename, symbol):
    spec = importlib.util.spec_from_file_location(filename, _here / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)

enhance         = _load("01_enhance.py",    "enhance")
segment         = _load("02_segment.py",    "segment")
get_hand_region = _load("03_contour.py",    "get_hand_region")
get_landmarks   = _load("04_landmarks.py",  "get_landmarks")
hand_present    = _load("05_background.py", "hand_present")
predict_letter  = _load("06_classify.py",   "predict_letter")
Smoother        = _load("07_smooth.py",     "Smoother")

_smoother = Smoother()


def reset_state():
    _smoother.reset()


def finalize_word():
    return _smoother.finalize_word()


def backspace():
    return _smoother.backspace()


def get_current_text():
    return _smoother.get_text()


def get_text_state():
    return {
        "word": _smoother.current_word,
        "sentence": _smoother.sentence,
        "full_text": _smoother.get_text(),
    }


def process_frame(frame, require_motion=True):
    # Step 1-2: enhance contrast then segment — overlay only, not a gate
    enhanced = enhance(frame)
    mask = segment(enhanced)

    # Step 3: draw contour overlay; bbox may be None but we continue regardless
    _, annotated = get_hand_region(mask, enhanced)

    # Step 4: skip classification if background subtraction sees no motion.
    # Feed "nothing" so an absent hand registers in the smoothing window — this is
    # what arms the next repeat (lets you sign the same letter twice deliberately).
    if require_motion and not hand_present(frame):
        _smoother.update("nothing")
        return annotated, None, None, None, _smoother.current_word, _smoother.sentence

    # Step 5: run MediaPipe on the full enhanced frame — no bbox crop needed
    h, w = enhanced.shape[:2]
    landmarks, annotated = get_landmarks(annotated, (0, 0, w, h))
    if landmarks is None:
        _smoother.update("nothing")
        return annotated, None, None, None, _smoother.current_word, _smoother.sentence

    # Step 6: predict ASL letter from landmarks
    result = predict_letter(landmarks)
    if result is None:
        _smoother.update("nothing")
        return annotated, None, None, None, _smoother.current_word, _smoother.sentence
    letter, confidence = result

    # Step 7: smooth over the rolling window and update the word/sentence builder
    # Every prediction is fed to the smoother regardless of confidence — the
    # majority-vote window is what filters out noisy/wrong guesses (that's its
    # job), so gating on confidence here only starves it of data and makes
    # commits take far longer to occur.
    committed, word, sentence = _smoother.update(letter)

    # Report the raw per-frame prediction as the detected letter for live, real-time
    # feedback while a sign is being formed. The smoother still gates what actually
    # gets committed into the word, so the word stays accurate.
    return annotated, committed, confidence, letter, word, sentence
