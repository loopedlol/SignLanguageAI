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
    HAND_LANDMARKER_MODEL_PATH,
)
from mediapipe_utils import create_hand_landmarker
from webcam_overlay import draw_text


WINDOW_NAME = "KSL HandLandmarker Debug Demo"


def _draw_hand_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    height, width, _ = frame.shape

    for hand in hand_landmarks:
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 3, (40, 220, 255), -1)


def main() -> int:
    model_path = HAND_LANDMARKER_MODEL_PATH
    landmarker = create_hand_landmarker(model_path)
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
            hand_landmarks = result.hand_landmarks or []
            _draw_hand_landmarks(frame, hand_landmarks)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                fps = 1.0 / elapsed

            draw_text(frame, f"hands detected: {len(hand_landmarks)}", 0)
            draw_text(frame, f"FPS: {fps:.1f}", 1)
            draw_text(frame, "Press q to quit", 2)

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
