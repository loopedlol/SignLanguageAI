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
    PROCESS_EVERY_N_FRAMES,
)
from feature_extractor import FEATURE_COUNT, extract_landmarks, flatten_landmarks
from mediapipe_utils import create_holistic_landmarker
from webcam_overlay import draw_all_landmarks, draw_landmark_debug_overlay, draw_text


WINDOW_NAME = "KSL MediaPipe Tasks Demo"


def main() -> int:
    landmarker = create_holistic_landmarker(MEDIAPIPE_MODEL_PATH)
    if landmarker is None:
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the default webcam.", file=sys.stderr)
        landmarker.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

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
    features = [0.0] * FEATURE_COUNT
    zero_ratio = 1.0

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
                features = flatten_landmarks(landmarks)
                zero_ratio = float(np.mean(np.asarray(features, dtype=np.float32) == 0))

                processing_elapsed = time.perf_counter() - processing_start
                if processing_elapsed > 0:
                    processing_fps = 1.0 / processing_elapsed

            draw_all_landmarks(frame, landmarks)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                display_fps = 1.0 / elapsed

            next_row = draw_landmark_debug_overlay(
                frame,
                landmarks,
                zero_ratio,
                frame_is_usable=None,
            )
            draw_text(frame, f"Display FPS: {display_fps:.1f}", next_row)
            draw_text(frame, f"MediaPipe FPS: {processing_fps:.1f}", next_row + 1)

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
