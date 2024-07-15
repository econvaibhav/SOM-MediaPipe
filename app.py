#!/usr/bin/env python3
"""SOM course gesture demo; adapted from Kazuhito Takahashi and Nikita Kiselov.

Hand landmarks, static poses, and optional fingertip motion. Apache-2.0.
"""
import argparse
import csv
from pathlib import Path
from utils.features import normalize_landmarks, normalize_history
from utils.motion import MotionState
from utils.recording import append_sample

ROOT = Path(__file__).resolve().parent


def confidence(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('Confidence must be between 0 and 1')
    return value


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument('--min_detection_confidence', type=confidence, default=0.7)
    parser.add_argument('--min_tracking_confidence', type=confidence, default=0.5)
    parser.add_argument('--max_num_hands', type=int, choices=[1, 2], default=1,
                        help='Two-hand mode classifies static signs only')
    parser.add_argument('--record_dir', type=Path, default=ROOT / 'data/recordings',
                        help='New CSV recordings; bundled datasets are never appended to')
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error('Width and height must be positive')
    return args


def labels(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [r[0] for r in csv.reader(stream) if r and r[0].strip()]


def select_mode(key, mode):
    number = key - 48 if 48 <= key <= 57 else -1
    return number, {ord('n'): 0, ord('k'): 1, ord('h'): 2}.get(key, mode)


def landmark_pixels(image, landmarks):
    height, width = image.shape[:2]
    return [[max(0, min(int(p.x * width), width - 1)),
             max(0, min(int(p.y * height), height - 1))] for p in landmarks.landmark]


def main(argv=None):
    args = get_args(argv)
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    from model import KeyPointClassifier, PointHistoryClassifier
    from utils.cvfpscalc import CvFpsCalc
    from utils.drawing import (draw_bounding_rect, draw_landmarks, draw_info_text,
                               draw_point_history, draw_info)

    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Use the legacy MediaPipe version in requirements.txt')
    signs = labels(ROOT / 'model/keypoint_classifier/keypoint_classifier_label.csv')
    motions = labels(ROOT / 'model/point_history_classifier/point_history_classifier_label.csv')
    sign_model, motion_model = KeyPointClassifier(), PointHistoryClassifier()
    if (len(signs), len(motions)) != (sign_model.class_count, motion_model.class_count):
        raise ValueError('Model output sizes must match the corresponding label CSV files')
    pointer_id = signs.index('Pointer')
    motion_enabled = args.max_num_hands == 1
    if not motion_enabled:
        print('Two-hand mode: static signs only. Use --max_num_hands 1 for finger motion.')
    print('n: normal | k: record sign | h: record motion | digit: save one sample | Esc: quit')
    print(f'New recordings: {args.record_dir.resolve()}')
    cap = cv.VideoCapture(args.device)
    state, mode, fps_calc = MotionState(16), 0, CvFpsCalc(buffer_len=10)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {args.device}; check device number and permission')
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        with mp.solutions.hands.Hands(
            static_image_mode=args.use_static_image_mode, max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as hands:
            while True:
                key = cv.waitKey(10) & 0xFF
                if key == 27:
                    break
                number, mode = select_mode(key, mode)
                if mode == 2 and not motion_enabled:
                    print('Motion recording needs --max_num_hands 1')
                    mode = 0
                ok, frame = cap.read()
                if not ok:
                    print('Camera frame unavailable; stopping.')
                    break
                frame = cv.flip(frame, 1)
                display = frame.copy()
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                if not result.multi_hand_landmarks:
                    state.reset()
                for landmarks, handedness in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
                    points = landmark_pixels(frame, landmarks)
                    x, y, w, h = cv.boundingRect(np.asarray(points, dtype=np.int32))
                    rect = [x, y, x + w, y + h]
                    features = normalize_landmarks(points)
                    sign_id = sign_model(features)
                    motion_text = ''
                    if motion_enabled:
                        state.update(handedness.classification[0].label, sign_id == pointer_id, points[8])
                        history = normalize_history(state.points, frame.shape[1], frame.shape[0])
                        if state.ready:
                            motion_text = motions[state.smooth(motion_model(history))]
                            if mode == 2 and number >= 0:
                                saved = append_sample(args.record_dir / 'point_history.csv', number,
                                                      history, feature_count=32, class_count=10)
                                print(f'Saved motion sample for class {number}' if saved else 'Invalid sample')
                        elif mode == 2 and number >= 0:
                            print('Wait for 16 consecutive Pointer frames before recording motion')
                    if mode == 1 and number >= 0:
                        saved = append_sample(args.record_dir / 'keypoint.csv', number,
                                              features, feature_count=42, class_count=10)
                        print(f'Saved sign sample for class {number}' if saved else 'Invalid sample')
                    display = draw_bounding_rect(True, display, rect)
                    display = draw_landmarks(display, points)
                    display = draw_info_text(display, rect, handedness, signs[sign_id], motion_text)
                display = draw_point_history(display, state.points)
                display = draw_info(display, fps_calc.get(), mode, number)
                cv.imshow('SOM course | MediaPipe gestures', display)
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
