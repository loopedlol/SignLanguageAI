# SignLanguageAI / Workflow guide

[← Project overview](../README.md) · [Evaluation boundaries](../README.md#evaluation) · [Demo brief](VISUALS.md)

Run all commands from the repository root. This guide preserves the detailed recording, training, and diagnostic workflow. Data, detector assets, and trained checkpoints must be supplied locally.

## Quick start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with:

```powershell
.venv\Scripts\activate
```

The MediaPipe task model is not committed to the repository. Download the Holistic Landmarker model and place it at:

```text
models/holistic_landmarker.task
```

Then the normal experiment workflow is:

```bash
python src/record_landmark_sequence.py --label hello --seconds 2
python src/inspect_dataset.py
python src/normalize_landmarks.py
python src/train.py
python src/evaluate.py
python src/predict_webcam.py
```

## Project structure

```text
README.md
requirements.txt
models/
  holistic_landmarker.task
src/
  config.py
  webcam_mediapipe_demo.py
  hand_landmarker_demo.py
  holistic_health_check.py
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
  run_holistic_health_check.sh
  train_30.sh
  evaluate_30.sh
  predict_30.sh
data/
  raw_videos/
  processed_landmarks/
  normalized_landmarks/
```

Most default paths and runtime settings live in `src/config.py`, including sequence length, checkpoint folder, camera settings, prediction threshold, and training settings.

## 1. Test landmark detection

Run the main webcam preprocessing demo:

```bash
python src/webcam_mediapipe_demo.py
```

The demo opens the default webcam, runs the MediaPipe Tasks API Holistic Landmarker in video mode, extracts pose/face/hand landmarks, and overlays detections on the video. Press `q` to exit.

For a smaller hand-only diagnostic, place a Hand Landmarker model at:

```text
models/hand_landmarker.task
```

and run:

```bash
python src/hand_landmarker_demo.py
```

To inspect the full Holistic pipeline and landmark counts:

```bash
python src/holistic_health_check.py
```

## 2. Record training sequences

Record `.npy` landmark sequences for one label:

```bash
python src/record_landmark_sequence.py --label hello --seconds 2
```

Example with optional arguments:

```bash
python src/record_landmark_sequence.py \
  --label thank_you \
  --seconds 2 \
  --output-dir data/processed_landmarks \
  --process-every-n-frames 1
```

When the preview opens:

- press `r` to record one sample;
- press `q` to quit;
- after saving, the recorder returns to preview mode so another sample can be captured for the same label.

Samples are stored by label:

```text
data/processed_landmarks/
  hello/
    hello_001.npy
    hello_002.npy
```

Each sample is shaped as `frames x features`. Missing landmark groups are zero-filled so every frame uses a consistent feature vector.

## 3. Inspect and normalize the dataset

Inspect recorded data before training:

```bash
python src/inspect_dataset.py
```

Optional example:

```bash
python src/inspect_dataset.py \
  --data-dir data/processed_landmarks \
  --expected-features 1659 \
  --min-samples 10
```

The inspector reports label counts, shapes, frame-length statistics, invalid files, mostly-zero samples, and underrepresented labels.

Normalize the landmark sequences with:

```bash
python src/normalize_landmarks.py
```

The normalizer preserves the label-folder structure. Pose shoulders are used as the body reference: landmarks are centered around the shoulder midpoint and scaled by shoulder width, while all-zero missing groups remain zero.

## 4. Load data for PyTorch

Preview the normalized data through the project dataset class:

```bash
python src/dataset.py
```

or specify settings explicitly:

```bash
python src/dataset.py --data-dir data/normalized_landmarks --sequence-length 30
```

`LandmarkSequenceDataset` creates stable label mappings, loads `.npy` sequences, and returns `(sequence, label)` tensors. Sequences are trimmed or zero-padded to `30 x 1659` by default.

## 5. Temporal CNN

Run a forward-pass check:

```bash
python src/model.py
```

The model accepts tensors shaped `batch x sequence_length x input_features`, transposes them for `Conv1d`, and returns raw class logits shaped `batch x num_classes`. Softmax is intentionally omitted from the model because training uses a classification loss on the raw logits.

## 5. Train the classifier

Train the isolated-sign Temporal CNN:

```bash
python src/train.py
```

Useful optional arguments include:

```text
--sequence-length 30
--epochs 50
--batch-size 4
--lr 0.001
--checkpoint-dir checkpoints_30
--val-split 0.2
--dropout 0.3
--seed 42
```

The script automatically uses CUDA, Apple Silicon MPS, or CPU and saves:

```text
checkpoints_30/latest.pt
checkpoints_30/best.pt
checkpoints_30/label_mapping.json
```

## 6. Evaluate a checkpoint

The default command evaluates every sample in the normalized directory, including training samples. It is a diagnostic, not independent held-out accuracy. For an independent check, use a separate normalized directory with exactly the same label set and ordering as the checkpoint; the evaluator warns about a mapping mismatch but does not remap labels.

Evaluate a trained model with:

```bash
python src/evaluate.py
```

or explicitly:

```bash
python src/evaluate.py \
  --data-dir data/normalized_landmarks \
  --checkpoint checkpoints_30/best.pt \
  --sequence-length 30
```

The evaluator reports total accuracy, per-class accuracy, incorrect predictions with confidence, and a plain-text confusion matrix. Add `--show-correct` to print correct predictions as well.

## 7. Run live prediction

Use a trained checkpoint for webcam inference:

```bash
python src/predict_webcam.py
```

Optional arguments include `--camera-index 0`, `--process-every-n-frames 1`, `--confidence-threshold 0.65`, and `--prediction-interval 3`.

The prediction script keeps a rolling landmark buffer, applies the same preprocessing used for training, runs the Temporal CNN, smooths recent predictions, and overlays the result on the webcam feed.

## Model files: `.task` vs `.pt`

The project uses two different model formats:

- MediaPipe `.task`: the landmark detector used to extract pose, face, and hand points from frames;
- PyTorch `.pt`: the trained sign classifier checkpoint produced by `src/train.py`.

The prediction script checks these paths so the two model types are not accidentally swapped.

## Convenience scripts

Several shell scripts wrap common commands:

```bash
scripts/run_demo.sh
scripts/run_holistic_health_check.sh
scripts/train_30.sh
scripts/evaluate_30.sh
scripts/predict_30.sh
```

Generated data and checkpoints can be reset with:

```bash
bash scripts/clear_data.sh
bash scripts/clear_models.sh
bash scripts/clear_all.sh
```

These cleanup scripts do not delete the MediaPipe `.task` model files.

## Troubleshooting MediaPipe

If `hand_landmarker_demo.py` works but the Holistic pipeline does not:

1. run `python src/holistic_health_check.py`;
2. verify or re-download `models/holistic_landmarker.task`;
3. if Holistic still fails, consider separating the main pipeline into `HandLandmarker` and `PoseLandmarker` components.

## Current limitations

- This is an isolated-sign classifier, not continuous sign-language translation.
- Model quality depends heavily on the amount and diversity of recorded training data.
- Landmark extraction quality can vary with camera placement, lighting, occlusion, and motion.
- The current feature vector is large because it includes pose, face, and both hands.
- A stronger evaluation setup would use a larger held-out dataset and more systematic per-class analysis.

## Next steps

The next useful milestone is to make isolated-sign recognition more reliable with more varied data, stronger held-out evaluation, and better per-class diagnostics. After that, the project could explore longer temporal context and eventually continuous or sentence-level recognition.
