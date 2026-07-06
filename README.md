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
  config.py
  webcam_mediapipe_demo.py
  hand_landmarker_demo.py
  record_landmark_sequence.py
  inspect_dataset.py
  normalize_landmarks.py
  dataset.py
  model.py
  train.py
  evaluate.py
  predict_webcam.py
  feature_extractor.py
scripts/
  run_demo.sh
  train_30.sh
  evaluate_30.sh
  predict_30.sh
data/
  raw_videos/
  processed_landmarks/
  normalized_landmarks/
```

The `models/holistic_landmarker.task` file is not committed. Download the
MediaPipe Holistic Landmarker task model and place it at that path before
running the demo.

Most default paths and runtime settings live in `src/config.py`. You can edit
that file to change the default sequence length, checkpoint folder, camera
settings, prediction threshold, and training settings without typing long
terminal commands.

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

Important file distinction:

- MediaPipe `.task` file: the landmark detector used to extract body, face, and
  hand points from webcam frames.
- PyTorch `.pt` file: the trained KSL sign classifier checkpoint saved by
  `src/train.py`.

Do not swap these paths. The prediction script checks that `--model-path` points
to a `.task` file and `--checkpoint` points to a `.pt` file.

## Run the Webcam Demo

```bash
python src/webcam_mediapipe_demo.py
```

Press `q` while the OpenCV window is focused to exit cleanly.

## Debug Hand Detection Only

To test MediaPipe hand landmark detection independently from Holistic and the
Temporal CNN, download the MediaPipe Hand Landmarker model to:

```text
models/hand_landmarker.task
```

Then run:

```bash
python src/hand_landmarker_demo.py
```

This demo uses only the MediaPipe Tasks API `HandLandmarker`, draws hand
landmark dots, displays `hands detected: 0/1/2`, and exits with `q`.

## Record Landmark Sequences

Use the recorder to collect `.npy` training samples for a sign label:

```bash
python src/record_landmark_sequence.py --label hello --seconds 2
```

Optional arguments:

```bash
python src/record_landmark_sequence.py \
  --label thank_you \
  --seconds 2 \
  --output-dir data/processed_landmarks \
  --process-every-n-frames 1
```

When the preview window opens:

- Press `r` to record one sample.
- Press `q` to quit.
- After a sample is saved, the script returns to preview mode so you can record
  another sample with the same label.

Samples are saved under a label folder with auto-incrementing names:

```text
data/
  processed_landmarks/
    hello/
      hello_001.npy
      hello_002.npy
      hello_003.npy
```

Each saved file is a NumPy array with shape:

```text
frames x features
```

Missing landmark groups are zero-filled so every frame has the same feature
length. For the current feature layout, the vector contains pose, face, left
hand, and right hand landmarks as `x, y, z` values.

## Inspect the Dataset

Before training, inspect the recorded `.npy` files:

```bash
python src/inspect_dataset.py
```

Optional arguments:

```bash
python src/inspect_dataset.py \
  --data-dir data/processed_landmarks \
  --expected-features 1659 \
  --min-samples 10
```

The inspector reports label counts, sample shapes, frame length statistics,
invalid files, mostly-zero files, and labels with too few samples.

## Normalize Landmark Sequences

Normalize recorded samples before model training:

```bash
python src/normalize_landmarks.py
```

The normalizer preserves the label folder structure:

```text
data/processed_landmarks/hello/hello_001.npy
data/normalized_landmarks/hello/hello_001.npy
```

Each frame keeps the same `1659` feature length. Pose shoulders are used as the
body reference: landmarks are centered around the shoulder midpoint and scaled
by shoulder width. All-zero missing landmark groups remain zero.

## Load Data for PyTorch

Preview the normalized landmark dataset as a PyTorch `Dataset`:

```bash
python src/dataset.py
```

Optional arguments:

```bash
python src/dataset.py --data-dir data/normalized_landmarks --sequence-length 30
```

`LandmarkSequenceDataset` reads label folders alphabetically, creates stable
label mappings, loads `.npy` sequences, and returns `(sequence, label)` tensors.
Sequences are trimmed or zero-padded to `30 x 1659` by default. This prepares
the data shape for a future Temporal CNN, but does not train a model yet.

## Test the Temporal CNN

Run a forward-pass check for the first Temporal CNN model:

```bash
python src/model.py
```

The model accepts tensors shaped `batch x sequence_length x input_features`,
transposes them for `Conv1d`, and returns raw class logits shaped
`batch x num_classes`. It intentionally does not apply softmax; a future
training script should use `CrossEntropyLoss`.

## Train the Prototype Classifier

Train the first isolated-sign Temporal CNN on normalized landmarks:

```bash
python src/train.py
```

Optional arguments include `--sequence-length 30`, `--epochs 50`,
`--batch-size 4`, `--lr 0.001`, `--checkpoint-dir checkpoints_30`,
`--val-split 0.2`, `--dropout 0.3`, and `--seed 42`.
The script automatically uses CUDA, Apple Silicon MPS, or CPU. It saves:

```text
checkpoints_30/latest.pt
checkpoints_30/best.pt
checkpoints_30/label_mapping.json
```

This is an early training prototype for isolated signs only.

## Evaluate a Checkpoint

Evaluate a trained checkpoint on normalized landmark samples:

```bash
python src/evaluate.py \
  --data-dir data/normalized_landmarks \
  --checkpoint checkpoints_30/best.pt \
  --sequence-length 30
```

With the defaults in `src/config.py`, this is usually enough:

```bash
python src/evaluate.py
```

Add `--show-correct` to also print correctly classified samples. The evaluator
prints total accuracy, per-class accuracy, incorrect predictions with
confidence, and a plain-text confusion matrix.

## Run Live Prediction

Use a trained checkpoint with the webcam for isolated-sign prediction:

```bash
python src/predict_webcam.py
```

Optional arguments include `--camera-index 0`, `--process-every-n-frames 1`,
`--confidence-threshold 0.65`, and `--prediction-interval 3`. The script keeps
a rolling landmark buffer, normalizes it with the same preprocessing used for
training, runs the Temporal CNN, smooths recent predictions, and displays the
result on the webcam feed.

## Normal Workflow

The common workflow now uses short commands:

```bash
python src/record_landmark_sequence.py --label hello --seconds 2
python src/inspect_dataset.py
python src/normalize_landmarks.py
python src/train.py
python src/evaluate.py
python src/predict_webcam.py
```

Optional convenience shell scripts are also available:

```bash
scripts/run_demo.sh
scripts/train_30.sh
scripts/evaluate_30.sh
scripts/predict_30.sh
```

## What the Demo Does

- Opens the default webcam with OpenCV.
- Converts frames from BGR to RGB.
- Wraps each RGB frame as an `mp.Image`.
- Runs the MediaPipe Tasks API Holistic Landmarker in video mode.
- Extracts pose, face, left hand, and right hand landmarks.
- Draws detected landmark points directly with OpenCV.
- Displays detection status for pose, face, left hand, and right hand.
- Displays camera/display FPS and MediaPipe processing FPS.
- Limits webcam capture to 640x480 at 30 FPS. Processing cadence is controlled
  by `PROCESS_EVERY_N_FRAMES` in `src/config.py`.

Landmark extraction helpers live in `src/feature_extractor.py`. Drawing helpers
use variable-length landmark groups, while saved training samples use
fixed-length zero-filled feature vectors.

To tune performance, change `PROCESS_EVERY_N_FRAMES` in
`src/webcam_mediapipe_demo.py`. Higher values reduce MediaPipe work but make
landmarks update less often.

## Next Milestone

After the isolated-sign prototype trains reliably, add evaluation tooling and
real-time prediction before attempting sentence-level translation.
