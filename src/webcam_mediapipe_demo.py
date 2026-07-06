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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extractor import extract_landmarks


WINDOW_NAME = "KSL MediaPipe Tasks Demo"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "holistic_landmarker.task"
PROCESS_EVERY_N_FRAMES = 2


def _print_model_error() -> None:
    print(
        "\nMissing MediaPipe model file.\n"
        f"Expected: {MODEL_PATH}\n\n"
        "Download the MediaPipe Holistic Landmarker .task model and place it at "
        "that path before running the webcam demo.\n",
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
            "Tasks API, or use separate Tasks API landmarkers for pose, face, "
            "and hands.\n",
            file=sys.stderr,
        )
        return None

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _draw_status(frame: np.ndarray, label: str, detected: bool, row: int) -> None:
    status = "detected" if detected else "missing"
    color = (40, 200, 40) if detected else (40, 40, 220)
    y = 30 + row * 28

    cv2.putText(
        frame,
        f"{label}: {status}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_fps(frame: np.ndarray, display_fps: float, processing_fps: float) -> None:
    cv2.putText(
        frame,
        f"Display FPS: {display_fps:.1f}",
        (10, 30 + 4 * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"MediaPipe FPS: {processing_fps:.1f}",
        (10, 30 + 5 * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
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


def main() -> int:
    landmarker = _create_landmarker()
    if landmarker is None:
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the default webcam.", file=sys.stderr)
        landmarker.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    previous_time = time.perf_counter()
    previous_timestamp_ms = 0
    display_fps = 0.0
    processing_fps = 0.0
    frame_count = 0
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
            should_process = frame_count % PROCESS_EVERY_N_FRAMES == 1

            if should_process:
                processing_start = time.perf_counter()
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

                processing_elapsed = time.perf_counter() - processing_start
                if processing_elapsed > 0:
                    processing_fps = 1.0 / processing_elapsed

            _draw_all_landmarks(frame, landmarks)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                display_fps = 1.0 / elapsed

            _draw_status(frame, "Pose", len(landmarks["pose"]) > 0, 0)
            _draw_status(frame, "Face", len(landmarks["face"]) > 0, 1)
            _draw_status(frame, "Left hand", len(landmarks["left_hand"]) > 0, 2)
            _draw_status(frame, "Right hand", len(landmarks["right_hand"]) > 0, 3)
            _draw_fps(frame, display_fps, processing_fps)

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
