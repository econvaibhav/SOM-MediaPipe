"""One-hand motion state; reset on loss or a change in the detected hand."""
from collections import Counter, deque


class MotionState:
    def __init__(self, length=16):
        self.length = length
        self.points = deque(maxlen=length)
        self.predictions = deque(maxlen=length)
        self.hand = None

    def reset(self):
        self.points.clear()
        self.predictions.clear()
        self.hand = None

    def update(self, hand, is_pointer, tip):
        if not is_pointer:
            self.reset()
            return
        if hand != self.hand:
            self.reset()
            self.hand = hand
        self.points.append(list(tip))

    @property
    def ready(self):
        return len(self.points) == self.length

    def smooth(self, label):
        self.predictions.append(label)
        return Counter(self.predictions).most_common(1)[0][0]
