"""Float32 TensorFlow Lite wrapper derived from the upstream classifiers."""
from pathlib import Path
import numpy as np
import tensorflow as tf


class LiteClassifier:
    def __init__(self, model_path, feature_count, num_threads=1):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f'Model not found: {model_path}')
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.feature_count = feature_count
        spec = self.input_details[0]
        if list(spec['shape']) != [1, feature_count] or spec['dtype'] != np.float32:
            raise ValueError(f'Expected a float32 [1, {feature_count}] input')
        self.class_count = int(self.output_details[0]['shape'][-1])

    def scores(self, features):
        data = np.asarray(features, dtype=np.float32)
        if data.shape != (self.feature_count,) or not np.isfinite(data).all():
            raise ValueError(f'Expected {self.feature_count} finite features')
        self.interpreter.set_tensor(self.input_details[0]['index'], data[None, :])
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        if not np.isfinite(scores).all():
            raise ValueError('Model returned non-finite scores')
        return scores
