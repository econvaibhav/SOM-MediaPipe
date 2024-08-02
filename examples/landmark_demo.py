"""Hand/face examples adapted from Google's MediaPipe solution samples (Apache-2.0).

Maintenance: command line, image output, camera errors, and resource cleanup.
"""
import argparse
from pathlib import Path


def main(task):
    parser = argparse.ArgumentParser(description=f'MediaPipe {task} landmark demo')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--image', type=Path, help='Use one image instead of a webcam')
    parser.add_argument('--output', type=Path, default=Path('artifacts') / f'{task}-preview.png')
    args = parser.parse_args()
    import cv2 as cv
    import mediapipe as mp
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    solution = mp.solutions.hands if task == 'hands' else mp.solutions.face_mesh
    detector = solution.Hands(static_image_mode=bool(args.image), max_num_hands=2) if task == 'hands' else solution.FaceMesh(static_image_mode=bool(args.image), max_num_faces=1, refine_landmarks=True)

    def annotate(frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = detector.process(rgb)
        landmarks = result.multi_hand_landmarks if task == 'hands' else result.multi_face_landmarks
        for points in landmarks or []:
            if task == 'hands':
                drawing.draw_landmarks(frame, points, solution.HAND_CONNECTIONS,
                                       styles.get_default_hand_landmarks_style(),
                                       styles.get_default_hand_connections_style())
            else:
                for connections, style in [
                    (solution.FACEMESH_TESSELATION, styles.get_default_face_mesh_tesselation_style()),
                    (solution.FACEMESH_CONTOURS, styles.get_default_face_mesh_contours_style()),
                    (solution.FACEMESH_IRISES, styles.get_default_face_mesh_iris_connections_style()),
                ]:
                    drawing.draw_landmarks(frame, points, connections,
                                           landmark_drawing_spec=None, connection_drawing_spec=style)
        return frame

    cap = None
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
            cap = cv.VideoCapture(args.device)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera {args.device}')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cv.imshow(f'MediaPipe {task}', annotate(cv.flip(frame, 1)))
                if cv.waitKey(5) & 0xFF == 27:
                    break
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()
