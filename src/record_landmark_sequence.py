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

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    MEDIAPIPE_MODEL_PATH,
    PROCESSED_LANDMARKS_DIR,
    PROJECT_ROOT,
    PROCESS_EVERY_N_FRAMES,
)
from feature_extractor import FEATURE_COUNT, extract_landmarks, flatten_landmarks
from mediapipe_utils import create_holistic_landmarker
from webcam_overlay import (
    compute_landmark_debug,
    draw_all_landmarks,
    draw_landmark_debug_overlay,
    draw_text,
)


WINDOW_NAME = "KSL Landmark Sequence Recorder"


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
        default=PROCESSED_LANDMARKS_DIR,
        help="Directory where label subfolders and .npy samples are saved.",
    )
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=PROCESS_EVERY_N_FRAMES,
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


def _open_camera() -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
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

    landmarker = create_holistic_landmarker(MEDIAPIPE_MODEL_PATH)
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
    warning_message = ""
    warning_message_until = 0.0
    sequence: list[list[float]] = []
    current_features = [0.0] * FEATURE_COUNT
    current_zero_ratio = 1.0
    current_frame_is_usable = False
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
            should_process = (frame_count - 1) % args.process_every_n_frames == 0

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
                current_features = flatten_landmarks(landmarks)
                debug = compute_landmark_debug(landmarks, current_features)
                current_zero_ratio = debug["zero_ratio"]
                current_frame_is_usable = debug["frame_is_usable"]
            else:
                current_frame_is_usable = False

            if is_recording and current_frame_is_usable:
                sequence.append(current_features)
            if is_recording:
                elapsed = time.perf_counter() - recording_started_at
                if elapsed >= args.seconds:
                    sample_path = _next_sample_path(output_dir, label)
                    saved_array = np.asarray(sequence, dtype=np.float32)
                    if saved_array.size == 0:
                        saved_array = saved_array.reshape(0, FEATURE_COUNT)
                    np.save(sample_path, saved_array)
                    sample_zero_ratio = (
                        1.0 if saved_array.size == 0 else float(np.mean(saved_array == 0))
                    )
                    saved_message = f"Saved: {_display_path(sample_path)}"
                    saved_message_until = time.perf_counter() + 3.0
                    print(
                        f"{saved_message} shape=({len(sequence)}, {FEATURE_COUNT}) "
                        f"zero_ratio={sample_zero_ratio:.2f}"
                    )
                    if sample_zero_ratio > 0.90:
                        warning_message = "Warning: saved sample is mostly zeros"
                        warning_message_until = time.perf_counter() + 5.0
                        print(warning_message)
                    sequence = []
                    is_recording = False

            draw_all_landmarks(frame, landmarks)
            next_row = draw_landmark_debug_overlay(
                frame,
                landmarks,
                current_zero_ratio,
                current_frame_is_usable,
            )

            if is_recording:
                draw_text(frame, f"Recording: {label}", next_row, (40, 40, 220))
                draw_text(
                    frame,
                    f"Usable frames collected: {len(sequence)} / {target_frames}",
                    next_row + 1,
                    (40, 40, 220),
                )
            else:
                draw_text(frame, "Press r to start recording", next_row)
                draw_text(frame, "Press q to quit", next_row + 1)
                draw_text(frame, f"Label: {label}", next_row + 2)
                draw_text(frame, f"Feature count: {FEATURE_COUNT}", next_row + 3)
                if saved_message and time.perf_counter() < saved_message_until:
                    draw_text(frame, saved_message, next_row + 4, (40, 200, 40))
                if warning_message and time.perf_counter() < warning_message_until:
                    draw_text(frame, warning_message, next_row + 5, (40, 40, 220))

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
