from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ksl_mpl_cache"))

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    BEST_CHECKPOINT_PATH,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONFIDENCE_THRESHOLD,
    MEDIAPIPE_MODEL_PATH,
    PREDICTION_INTERVAL,
    PROCESS_EVERY_N_FRAMES,
    PROJECT_ROOT,
    SEQUENCE_LENGTH,
)
from feature_extractor import extract_landmarks, flatten_landmarks
from model import TemporalCNN
from normalize_landmarks import normalize_sequence
from train import get_device


WINDOW_NAME = "KSL Live Prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live KSL webcam prediction.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_CHECKPOINT_PATH,
        help="Path to a trained TemporalCNN checkpoint.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MEDIAPIPE_MODEL_PATH,
        help="Path to the MediaPipe Holistic Landmarker .task file.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Rolling buffer length.",
    )
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--process-every-n-frames", type=int, default=PROCESS_EVERY_N_FRAMES)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--prediction-interval", type=int, default=PREDICTION_INTERVAL)
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _normalize_index_to_label(index_to_label: dict) -> dict[int, str]:
    return {int(index): label for index, label in index_to_label.items()}


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict | None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"Checkpoint does not exist: {_display_path(checkpoint_path)}", file=sys.stderr)
        return None

    return torch.load(checkpoint_path, map_location=device)


def _create_model(checkpoint: dict, device: torch.device) -> TemporalCNN:
    model = TemporalCNN(
        input_features=int(checkpoint["input_features"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _create_landmarker(model_path: Path):
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"MediaPipe model does not exist: {_display_path(model_path)}", file=sys.stderr)
        return None

    if not hasattr(vision, "HolisticLandmarker") or not hasattr(
        vision, "HolisticLandmarkerOptions"
    ):
        print(
            "This installed MediaPipe package does not expose "
            "vision.HolisticLandmarker.",
            file=sys.stderr,
        )
        return None

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _open_camera(camera_index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open webcam at index {camera_index}.", file=sys.stderr)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap


def _draw_text(
    frame: np.ndarray,
    text: str,
    row: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        frame,
        text,
        (10, 30 + row * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_landmark_points(
    frame: np.ndarray,
    landmarks: list[list[float]],
    color: tuple[int, int, int],
) -> None:
    height, width, _ = frame.shape

    for x_norm, y_norm, _z_norm in landmarks:
        x = int(x_norm * width)
        y = int(y_norm * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 2, color, -1)


def _draw_all_landmarks(frame: np.ndarray, landmarks: dict[str, list[list[float]]]) -> None:
    _draw_landmark_points(frame, landmarks["pose"], (80, 220, 120))
    _draw_landmark_points(frame, landmarks["face"], (80, 110, 255))
    _draw_landmark_points(frame, landmarks["left_hand"], (255, 180, 90))
    _draw_landmark_points(frame, landmarks["right_hand"], (90, 220, 255))


def _predict_sequence(
    model: TemporalCNN,
    sequence_buffer: deque[list[float]],
    device: torch.device,
) -> tuple[int, float]:
    sequence = np.asarray(sequence_buffer, dtype=np.float32)
    normalized_sequence = normalize_sequence(sequence)
    tensor = torch.as_tensor(normalized_sequence, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    confidence, predicted_index = torch.max(probabilities, dim=0)
    return int(predicted_index.item()), float(confidence.item())


def _smooth_prediction(
    prediction_history: deque[tuple[str, float]],
    confidence_threshold: float,
) -> tuple[str, float]:
    confident_labels = [
        label for label, confidence in prediction_history
        if confidence >= confidence_threshold
    ]
    if not confident_labels:
        return "unsure", 0.0

    label_counts = Counter(confident_labels)
    best_label, best_count = label_counts.most_common(1)[0]
    tied_labels = [
        label for label, count in label_counts.items()
        if count == best_count
    ]
    if len(tied_labels) > 1:
        return "unsure", 0.0

    best_confidence = max(
        confidence for label, confidence in prediction_history
        if label == best_label
    )
    return best_label, best_confidence


def main() -> int:
    args = parse_args()

    if args.process_every_n_frames <= 0:
        print("--process-every-n-frames must be at least 1.", file=sys.stderr)
        return 1
    if args.prediction_interval <= 0:
        print("--prediction-interval must be at least 1.", file=sys.stderr)
        return 1
    if not 0 <= args.confidence_threshold <= 1:
        print("--confidence-threshold must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.model_path.suffix != ".task":
        print("Error: --model-path should point to a MediaPipe .task file.", file=sys.stderr)
        return 1
    if args.checkpoint.suffix != ".pt":
        print("Error: --checkpoint should point to a PyTorch .pt checkpoint.", file=sys.stderr)
        return 1

    device = get_device()
    checkpoint = _load_checkpoint(args.checkpoint, device)
    if checkpoint is None:
        return 1

    checkpoint_sequence_length = int(checkpoint["sequence_length"])
    sequence_length = args.sequence_length or checkpoint_sequence_length
    input_features = int(checkpoint["input_features"])
    index_to_label = _normalize_index_to_label(checkpoint.get("index_to_label", {}))
    model = _create_model(checkpoint, device)

    landmarker = _create_landmarker(args.model_path)
    if landmarker is None:
        return 1

    cap = _open_camera(args.camera_index)
    if cap is None:
        landmarker.close()
        return 1

    sequence_buffer: deque[list[float]] = deque(maxlen=sequence_length)
    prediction_history: deque[tuple[str, float]] = deque(maxlen=args.prediction_interval)
    landmarks = {
        "pose": [],
        "face": [],
        "left_hand": [],
        "right_hand": [],
    }
    frame_count = 0
    processed_frame_count = 0
    previous_timestamp_ms = 0
    raw_prediction = "waiting"
    raw_confidence = 0.0
    smoothed_prediction = "unsure"
    smoothed_confidence = 0.0

    try:
        while True:
            frame_count += 1
            success, frame = cap.read()
            if not success:
                print("Could not read a frame from the webcam.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            should_process = frame_count % args.process_every_n_frames == 1

            if should_process:
                processed_frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_frame),
                )
                timestamp_ms = int(time.perf_counter() * 1000)
                if timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = previous_timestamp_ms + 1
                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                landmarks = extract_landmarks(result)
                features = flatten_landmarks(landmarks)

                if len(features) != input_features:
                    print(
                        f"Feature count mismatch: got {len(features)}, "
                        f"expected {input_features}.",
                        file=sys.stderr,
                    )
                    break

                sequence_buffer.append(features)

                if len(sequence_buffer) == sequence_length:
                    predicted_index, raw_confidence = _predict_sequence(
                        model,
                        sequence_buffer,
                        device,
                    )
                    raw_prediction = index_to_label.get(
                        predicted_index,
                        str(predicted_index),
                    )
                    prediction_history.append((raw_prediction, raw_confidence))
                    smoothed_prediction, smoothed_confidence = _smooth_prediction(
                        prediction_history,
                        args.confidence_threshold,
                    )

            _draw_all_landmarks(frame, landmarks)

            display_prediction = (
                smoothed_prediction
                if smoothed_confidence >= args.confidence_threshold
                else "unsure"
            )
            prediction_color = (
                (40, 200, 40)
                if display_prediction != "unsure"
                else (40, 40, 220)
            )
            _draw_text(frame, f"Prediction: {display_prediction}", 0, prediction_color)
            _draw_text(frame, f"Confidence: {smoothed_confidence:.2f}", 1, prediction_color)
            _draw_text(frame, f"buffer: {len(sequence_buffer)}/{sequence_length}", 2)
            _draw_text(frame, f"device: {device}", 3)
            _draw_text(frame, f"raw prediction: {raw_prediction}", 4)
            _draw_text(frame, f"raw confidence: {raw_confidence:.2f}", 5)
            _draw_text(frame, f"smoothed prediction: {smoothed_prediction}", 6)
            _draw_text(frame, f"processed frames: {processed_frame_count}", 7)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
