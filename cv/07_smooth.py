# Task 7 - Temporal Smoothing and Word Builder
# Smooths per-frame predictions over a rolling window to reduce flickering,
# and accumulates stable letters over time to build recognised words/sentences.

import time
from collections import deque


class Smoother:
    WINDOW = 15          # number of recent frames to consider
    THRESHOLD = 0.70     # fraction of frames that must agree to commit a letter
    PAUSE = 1.5          # seconds of silence before the current word is finalised

    def __init__(self):
        self.reset()

    def reset(self):
        # Clear all state — window, current word, and full sentence
        self._buffer = deque(maxlen=self.WINDOW)
        self._last_committed = None
        self._last_commit_time = None
        self.current_word = ""
        self.sentence = ""

    def update(self, letter):
        # Push the latest prediction into the rolling window
        self._buffer.append(letter)

        # Find the dominant letter and its fraction of the window
        if len(self._buffer) == self.WINDOW:
            counts = {}
            for l in self._buffer:
                counts[l] = counts.get(l, 0) + 1
            dominant, freq = max(counts.items(), key=lambda x: x[1])
            confident = (freq / self.WINDOW) >= self.THRESHOLD
        else:
            dominant, confident = None, False

        now = time.monotonic()
        committed = None

        if confident and dominant != self._last_committed:
            # Commit the letter — avoid repeating the same letter consecutively
            self.current_word += dominant
            self._last_committed = dominant
            self._last_commit_time = now
            committed = dominant
        elif not confident and self._last_commit_time is not None:
            # Flush the current word to the sentence after a pause
            if now - self._last_commit_time >= self.PAUSE and self.current_word:
                self.sentence = (self.sentence + " " + self.current_word).strip()
                self.current_word = ""
                self._last_committed = None
                self._last_commit_time = None

        return committed, self.current_word, self.sentence
