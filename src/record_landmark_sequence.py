from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ksl_mpl_cache"))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extractor import FEATURE_COUNT, extract_landmarks, flatten_landmarks


WINDOW_NAME = "KSL Landmark Sequence Recorder"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "holistic_landmarker.task"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_landmarks"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record MediaPipe landmark sequences as .npy training samples."
    )
    parser.add_argument("--label", required=True, help="Sign label, e.g. hello")
    parser.add_argument(
        "--seconds",
        required=True,
        type=float,
        help="How many seconds to record for each sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where label subfolders and .npy samples are saved.",
    )
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=2,
        help="Run MediaPipe every N frames and reuse the latest result between runs.",
    )
    return parser.parse_args()


def _sanitize_label(label: str) -> str:
    normalized = label.strip().lower().replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_,-]+", "", normalized)
    normalized = normalized.strip("_,-")
    if not normalized:
        raise ValueError("Label must contain at least one letter or number.")
    return normalized


def _print_model_error() -> None:
    print(
        "\nMissing MediaPipe model file.\n"
        f"Expected: {MODEL_PATH}\n\n"
        "Download the MediaPipe Holistic Landmarker .task model and place it at "
        "that path before running the recorder.\n",
        file=sys.stderr,
    )


def _create_landmarker():
    if not MODEL_PATH.exists():
        _print_model_error()
        return None

    if not hasattr(vision, "HolisticLandmarker") or not hasattr(
        vision, "HolisticLandmarkerOptions"
    ):
        print(
            "\nThis installed MediaPipe package does not expose "
            "vision.HolisticLandmarker.\n"
            "Install a MediaPipe version that includes the Holistic Landmarker "
            "Tasks API.\n",
            file=sys.stderr,
        )
        return None

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _next_sample_path(output_dir: Path, label: str) -> Path:
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = []
    pattern = re.compile(rf"^{re.escape(label)}_(\d+)\.npy$")
    for path in label_dir.glob(f"{label}_*.npy"):
        match = pattern.match(path.name)
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    return label_dir / f"{label}_{next_index:03d}.npy"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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
    frame: np.ndarray, landmarks: list[list[float]], color: tuple[int, int, int]
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


def _open_camera() -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the default webcam.", file=sys.stderr)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap


def main() -> int:
    args = parse_args()

    try:
        label = _sanitize_label(args.label)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.seconds <= 0:
        print("--seconds must be greater than 0.", file=sys.stderr)
        return 1

    if args.process_every_n_frames <= 0:
        print("--process-every-n-frames must be at least 1.", file=sys.stderr)
        return 1

    landmarker = _create_landmarker()
    if landmarker is None:
        return 1

    cap = _open_camera()
    if cap is None:
        landmarker.close()
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    target_frames = int(round(args.seconds * CAMERA_FPS))
    previous_timestamp_ms = 0
    frame_count = 0
    is_recording = False
    recording_started_at = 0.0
    saved_message = ""
    saved_message_until = 0.0
    sequence: list[list[float]] = []
    landmarks = {
        "pose": [],
        "face": [],
        "left_hand": [],
        "right_hand": [],
    }

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

            if is_recording:
                sequence.append(flatten_landmarks(landmarks))
                elapsed = time.perf_counter() - recording_started_at
                if elapsed >= args.seconds:
                    sample_path = _next_sample_path(output_dir, label)
                    np.save(sample_path, np.asarray(sequence, dtype=np.float32))
                    saved_message = f"Saved: {_display_path(sample_path)}"
                    saved_message_until = time.perf_counter() + 3.0
                    print(f"{saved_message} shape=({len(sequence)}, {FEATURE_COUNT})")
                    sequence = []
                    is_recording = False

            _draw_all_landmarks(frame, landmarks)

            if is_recording:
                _draw_text(frame, f"Recording: {label}", 0, (40, 40, 220))
                _draw_text(
                    frame,
                    f"Frames collected: {len(sequence)} / {target_frames}",
                    1,
                    (40, 40, 220),
                )
            else:
                _draw_text(frame, "Press r to start recording", 0)
                _draw_text(frame, "Press q to quit", 1)
                _draw_text(frame, f"Label: {label}", 2)
                _draw_text(frame, f"Feature count: {FEATURE_COUNT}", 3)
                if saved_message and time.perf_counter() < saved_message_until:
                    _draw_text(frame, saved_message, 4, (40, 200, 40))

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r") and not is_recording:
                sequence = []
                recording_started_at = time.perf_counter()
                is_recording = True
                saved_message = ""
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
