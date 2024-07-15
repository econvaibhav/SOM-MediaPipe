"""Static sign classifier, adapted from the upstream Apache-2.0 implementation."""
from pathlib import Path
import numpy as np
from model.lite_classifier import LiteClassifier


class KeyPointClassifier(LiteClassifier):
    def __init__(self, model_path=None, num_threads=1):
        super().__init__(model_path or Path(__file__).with_name('keypoint_classifier.tflite'),
                         feature_count=42, num_threads=num_threads)

    def __call__(self, landmark_list):
        return int(np.argmax(self.scores(landmark_list)))
