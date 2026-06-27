# Task 7 - Temporal Smoothing and Word Builder
# Smooths per-frame predictions over a rolling window to reduce flickering,
# then turns stable ASL symbols into words and sentences.

from collections import deque


class Smoother:
    WINDOW = 25
    THRESHOLD = 0.75
    # Consecutive "hand absent" frames (nothing/space) that count as the hand
    # leaving the sign. Short enough that a natural pause between two identical
    # letters unlocks the repeat; long enough that 1-2 frame flicker won't.
    RELEASE_FRAMES = 4

    def __init__(self):
        self.reset()

    def reset(self):
        self._buffer = deque(maxlen=self.WINDOW)
        self._last_committed = None
        # Counts consecutive hand-absent frames so we can tell a deliberate pause
        # apart from a steady hold. A repeat of the same letter is only allowed
        # after the hand has clearly left the sign — holding steady never repeats.
        self._absent_run = 0
        self.current_word = ""
        self.sentence = ""

    def _finalize_word(self):
        if self.current_word:
            self.sentence = (self.sentence + " " + self.current_word).strip()
            self.current_word = ""
        self._last_committed = None
        self._absent_run = 0

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

        # Release detection: track how long the hand has been absent. "nothing"
        # and "space" mean no active sign (the pipeline feeds "nothing" whenever
        # the hand is lowered or lost). A short run of these confirms the hand
        # left the sign, so we forget the last commit and clear the now-stale
        # window — letting the next sign commit fresh even if it's the same
        # letter. Any real sign resets the counter, so holding steady (or a brief
        # 1-2 frame flicker) never triggers a release.
        if letter in ("nothing", "space"):
            self._absent_run += 1
            if self._absent_run >= self.RELEASE_FRAMES and self._last_committed is not None:
                self._last_committed = None
                self._buffer.clear()
        else:
            self._absent_run = 0

        dominant = None
        confident = False
        if len(self._buffer) == self.WINDOW:
            counts = {}
            for symbol in self._buffer:
                counts[symbol] = counts.get(symbol, 0) + 1
            dominant, freq = max(counts.items(), key=lambda item: item[1])
            confident = (freq / self.WINDOW) >= self.THRESHOLD

        committed = None

        if confident:
            if dominant in ("nothing", "space"):
                pass
            elif dominant != self._last_committed:
                # New letter, or the same letter signed again after a release
                # (which reset _last_committed to None). Holding one sign steady
                # keeps dominant == _last_committed, so it never re-commits.
                if dominant == "del":
                    self.backspace()
                else:
                    self.current_word += dominant
                self._last_committed = dominant
                committed = dominant
        # Auto-finalization removed — word only ends when End Word button is pressed

        return committed, self.current_word, self.sentence
