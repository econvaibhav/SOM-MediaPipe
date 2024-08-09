"""Regression checks for the faults found during review. Standard library only."""
import csv
import tempfile
import unittest
from pathlib import Path
from app import get_args, select_mode
from utils.features import normalize_landmarks, normalize_history
from utils.motion import MotionState
from utils.recording import append_sample


class FeatureTests(unittest.TestCase):
    def test_translation_and_scale_invariance(self):
        points = [[i, i * 2] for i in range(21)]
        shifted = [[3*x + 71, 3*y - 30] for x, y in points]
        self.assertEqual(normalize_landmarks(points), normalize_landmarks(shifted))
        self.assertEqual(normalize_landmarks(points)[:2], [0, 0])
        self.assertEqual(max(map(abs, normalize_landmarks(points))), 1)

    def test_coincident_points_do_not_divide_by_zero(self):
        self.assertEqual(normalize_landmarks([[5, 5]] * 21), [0.0] * 42)

    def test_invalid_features_rejected(self):
        for data in [[], [[1, 2]] * 20, [[float('nan'), 0]] * 21]:
            with self.assertRaises(ValueError):
                normalize_landmarks(data)

    def test_motion_is_relative_and_uses_frame_dimensions(self):
        self.assertEqual(normalize_history([[10, 20], [60, 70]], 100, 200), [0, 0, .5, .25])
        self.assertEqual(normalize_history([], 100, 100), [])


class MotionTests(unittest.TestCase):
    def test_only_complete_pointer_history_is_ready(self):
        state = MotionState()
        for i in range(15):
            state.update('Left', True, [i, 1])
        self.assertFalse(state.ready)
        state.update('Left', True, [15, 1])
        self.assertTrue(state.ready)
        state.update('Left', False, [16, 1])
        self.assertFalse(state.ready)
        self.assertEqual(len(state.points), 0)

    def test_hand_switch_and_loss_clear_predictions(self):
        state = MotionState()
        for i in range(16):
            state.update('Left', True, [i, 1])
        state.smooth(2)
        state.update('Right', True, [90, 90])
        self.assertEqual(list(state.points), [[90, 90]])
        self.assertEqual(list(state.predictions), [])
        state.reset()
        self.assertIsNone(state.hand)


class RecordingTests(unittest.TestCase):
    def test_partial_history_and_bad_labels_do_not_create_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'history.csv'
            for label, values in [(0, [0]*30), (9, [0]*32), (-1, [0]*32), (0, [float('inf')]*32)]:
                self.assertFalse(append_sample(path, label, values, feature_count=32, class_count=4))
            self.assertFalse(path.exists())
            self.assertTrue(append_sample(path, 3, [0]*32, feature_count=32, class_count=4))
            with path.open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]), 33)


class ArgumentTests(unittest.TestCase):
    def test_decimal_tracking_confidence(self):
        self.assertEqual(get_args(['--min_tracking_confidence', '0.65']).min_tracking_confidence, .65)

    def test_modes_and_digits(self):
        self.assertEqual(select_mode(ord('k'), 0), (-1, 1))
        self.assertEqual(select_mode(ord('4'), 1), (4, 1))
        self.assertEqual(select_mode(ord('n'), 1), (-1, 0))


if __name__ == '__main__':
    unittest.main()
