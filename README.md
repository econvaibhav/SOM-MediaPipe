# SOM MediaPipe

Hand-gesture recognition developed for the **Self-Organizing Maps course taught by Krista Lagus at the University of Helsinki**.

The project uses MediaPipe hand landmarks and a small neural network to classify five gestures: `Open`, `Close`, `Pointer`, `OK`, and `hi`. The `hi` class adds 168 course examples to the original four-class dataset.

## Method
1. MediaPipe detects **21 hand landmarks**.
2. Wrist-relative, scale-normalized coordinates produce **42 features**.
3. An MLP (**42 → 20 → 10 → 5**) returns the gesture label and model scores.

## Browser demo

~~~bash
python3 -m http.server 8000 --directory demo
~~~

Open **http://localhost:8000**. The recorded examples work immediately. Camera mode runs the saved classifier locally in the browser; it needs internet access once to load MediaPipe.

## Python application

Python 3.12 is recommended:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
~~~

`Esc` closes the webcam window. Press `k`, then `0`–`4`, to record a labelled pose in `data/recordings/`; press `n` to return to prediction mode.

Train and test:

~~~bash
python train.py
python train.py --task point_history
python3 -m unittest discover -s tests
node tests/check_browser_model.mjs
~~~

Training output is written to `artifacts/` so the bundled course models are not overwritten.

## Notes

- The static-pose dataset contains **4,955 rows**.
- Browser examples are recorded dataset rows, not a held-out evaluation set.
- Displayed softmax scores are model outputs, not calibrated probabilities.
- Evaluation on unseen people and recording sessions remains the appropriate next step.

## Attribution and licence

Based on [Kazuhito Takahashi's MediaPipe hand-gesture project](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe), via Nikita Kiselov's English fork. Hand tracking uses [Google MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js).

The full [Apache License 2.0](LICENSE) is retained because this repository adapts Apache-licensed upstream code; redistribution requires including that licence.
