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
from webcam_overlay import draw_all_landmarks, draw_landmark_debug_overlay, draw_text


WINDOW_NAME = "KSL Holistic Health Check"


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

            draw_all_landmarks(frame, landmarks)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                fps = 1.0 / elapsed

            next_row = draw_landmark_debug_overlay(
                frame,
                landmarks,
                zero_ratio,
                frame_is_usable=None,
            )
            draw_text(
                frame,
                "counts: "
                f"pose={len(landmarks['pose'])} "
                f"face={len(landmarks['face'])} "
                f"left={len(landmarks['left_hand'])} "
                f"right={len(landmarks['right_hand'])}",
                next_row,
            )
            draw_text(frame, f"FPS: {fps:.1f}", next_row + 1)
            draw_text(frame, "Press q to quit", next_row + 2)

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
