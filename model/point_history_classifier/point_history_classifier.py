"""Motion classifier, adapted from the upstream Apache-2.0 implementation."""
from pathlib import Path
import numpy as np
from model.lite_classifier import LiteClassifier


class PointHistoryClassifier(LiteClassifier):
    def __init__(self, model_path=None, score_th=0.5, invalid_value=0, num_threads=1):
        super().__init__(model_path or Path(__file__).with_name('point_history_classifier.tflite'),
                         feature_count=32, num_threads=num_threads)
        if not 0 <= score_th <= 1 or not 0 <= invalid_value < self.class_count:
            raise ValueError('Invalid confidence threshold or fallback class')
        self.score_th, self.invalid_value = score_th, invalid_value

    def __call__(self, point_history):
        scores = self.scores(point_history)
        index = int(np.argmax(scores))
        return index if scores[index] >= self.score_th else self.invalid_value
