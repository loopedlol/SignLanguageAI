from __future__ import annotations

import os
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
)
from feature_extractor import extract_landmarks, flatten_landmarks
from mediapipe_utils import create_holistic_landmarker


WINDOW_NAME = "KSL Holistic Health Check"


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


def _draw_status(frame: np.ndarray, label: str, landmarks: list[list[float]], row: int) -> None:
    detected = len(landmarks) > 0
    color = (40, 200, 40) if detected else (40, 40, 220)
    status = "detected" if detected else "missing"
    _draw_text(frame, f"{label}: {status} ({len(landmarks)})", row, color)


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


def main() -> int:
    landmarker = create_holistic_landmarker(MEDIAPIPE_MODEL_PATH)
    if landmarker is None:
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open webcam at index {CAMERA_INDEX}.", file=sys.stderr)
        landmarker.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    previous_time = time.perf_counter()
    previous_timestamp_ms = 0
    fps = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read a frame from the webcam.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
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
            zero_ratio = float(np.mean(np.asarray(features, dtype=np.float32) == 0))

            _draw_all_landmarks(frame, landmarks)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                fps = 1.0 / elapsed

            _draw_status(frame, "Pose", landmarks["pose"], 0)
            _draw_status(frame, "Face", landmarks["face"], 1)
            _draw_status(frame, "Left hand", landmarks["left_hand"], 2)
            _draw_status(frame, "Right hand", landmarks["right_hand"], 3)
            _draw_text(frame, f"zero ratio: {zero_ratio:.2f}", 4)
            _draw_text(frame, f"FPS: {fps:.1f}", 5)
            _draw_text(frame, "Press q to quit", 6)

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
