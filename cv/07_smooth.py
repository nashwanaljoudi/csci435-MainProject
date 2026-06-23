# Task 7 - Temporal Smoothing and Word Builder
# Smooths per-frame predictions over a rolling window to reduce flickering,
# then turns stable ASL symbols into words and sentences.

import time
from collections import deque


class Smoother:
    WINDOW = 25
    THRESHOLD = 0.75
    REPEAT_GAP = 4.0

    def __init__(self):
        self.reset()

    def reset(self):
        self._buffer = deque(maxlen=self.WINDOW)
        self._last_committed = None
        self._last_commit_time = None
        self.current_word = ""
        self.sentence = ""

    def _finalize_word(self):
        if self.current_word:
            self.sentence = (self.sentence + " " + self.current_word).strip()
            self.current_word = ""
        self._last_committed = None
        self._last_commit_time = None

    def finalize_word(self):
        self._finalize_word()
        self._buffer.clear()
        return self.current_word, self.sentence

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence[:-1].rstrip()
        return self.current_word, self.sentence

    def get_text(self):
        return (self.sentence + " " + self.current_word).strip()

    def update(self, letter):
        self._buffer.append(letter)

        dominant = None
        confident = False
        if len(self._buffer) == self.WINDOW:
            counts = {}
            for symbol in self._buffer:
                counts[symbol] = counts.get(symbol, 0) + 1
            dominant, freq = max(counts.items(), key=lambda item: item[1])
            confident = (freq / self.WINDOW) >= self.THRESHOLD

        now = time.monotonic()
        committed = None

        if confident:
            if dominant == "nothing":
                pass
            elif dominant == "space":
                pass
            elif dominant == "del":
                if dominant != self._last_committed:
                    self.backspace()
                    self._last_committed = dominant
                    self._last_commit_time = now
                    committed = "del"
            else:
                can_repeat = (
                    dominant == self._last_committed
                    and self._last_commit_time is not None
                    and now - self._last_commit_time >= self.REPEAT_GAP
                )
                if dominant != self._last_committed or can_repeat:
                    self.current_word += dominant
                    self._last_committed = dominant
                    self._last_commit_time = now
                    committed = dominant
        # Auto-finalization removed — word only ends when End Word button is pressed

        return committed, self.current_word, self.sentence
