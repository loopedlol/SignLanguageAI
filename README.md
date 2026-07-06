# KSL Temporal CNN

First milestone for a Korean Sign Language recognition project. This version
does not train a model yet; it provides a live webcam preprocessing demo that
captures video, runs MediaPipe Tasks API landmark detection, and overlays the
detected body points on the webcam feed.

## Project Structure

```text
README.md
requirements.txt
models/
  holistic_landmarker.task
src/
  webcam_mediapipe_demo.py
  feature_extractor.py
data/
  raw_videos/
  processed_landmarks/
```

The `models/holistic_landmarker.task` file is not committed. Download the
MediaPipe Holistic Landmarker task model and place it at that path before
running the demo.

## Install

From this directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If your system exposes Python as `python3`, use `python3` for the commands
above instead.

## Model File

The webcam demo expects:

```text
models/holistic_landmarker.task
```

If that file is missing, the script prints a clear error and exits. The demo
uses the modern MediaPipe Tasks API via:

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

It does not use the old `mediapipe.solutions` API.

## Run the Webcam Demo

```bash
python src/webcam_mediapipe_demo.py
```

Press `q` while the OpenCV window is focused to exit cleanly.

## What the Demo Does

- Opens the default webcam with OpenCV.
- Converts frames from BGR to RGB.
- Wraps each RGB frame as an `mp.Image`.
- Runs the MediaPipe Tasks API Holistic Landmarker in video mode.
- Extracts pose, face, left hand, and right hand landmarks.
- Draws detected landmark points directly with OpenCV.
- Displays detection status for pose, face, left hand, and right hand.
- Displays camera/display FPS and MediaPipe processing FPS.
- Limits webcam capture to 640x480 at 30 FPS and processes every other frame by
  default to reduce heat and CPU/GPU load.

Landmark extraction helpers live in `src/feature_extractor.py`. Missing
landmark groups are represented as empty lists for this first milestone.

To tune performance, change `PROCESS_EVERY_N_FRAMES` in
`src/webcam_mediapipe_demo.py`. Higher values reduce MediaPipe work but make
landmarks update less often.

## Next Milestone

Add a small data collection and preprocessing workflow:

- Record short labeled webcam clips into `data/raw_videos/`.
- Save per-frame landmark dictionaries or flattened vectors into
  `data/processed_landmarks/`.
- Decide on padding/interpolation rules for missing landmark groups.
- Only after that, begin the first Temporal CNN training pipeline.
