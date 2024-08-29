"""Display MediaPipe hand landmarks from a webcam or image."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--image', type=Path, help='Use one image instead of a webcam')
    parser.add_argument('--output', type=Path, default=Path('artifacts/hand-landmarks.png'))
    args = parser.parse_args()

    import cv2 as cv
    import mediapipe as mp

    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    detector = mp.solutions.hands.Hands(
        static_image_mode=bool(args.image),
        max_num_hands=2,
    )

    def annotate(frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = detector.process(rgb)
        for points in result.multi_hand_landmarks or []:
            drawing.draw_landmarks(
                frame,
                points,
                mp.solutions.hands.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )
        return frame

    camera = None
    try:
        with detector:
            if args.image:
                frame = cv.imread(str(args.image))
                if frame is None:
                    raise ValueError(f'Cannot read image: {args.image}')
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if not cv.imwrite(str(args.output), annotate(frame)):
                    raise OSError(f'Could not write {args.output}')
                print(args.output.resolve())
                return

            camera = cv.VideoCapture(args.device)
            if not camera.isOpened():
                raise RuntimeError(f'Cannot open camera {args.device}')
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                cv.imshow('MediaPipe hand landmarks', annotate(cv.flip(frame, 1)))
                if cv.waitKey(5) & 0xFF == 27:
                    break
    finally:
        if camera is not None:
            camera.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
