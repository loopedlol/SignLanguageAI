from __future__ import annotations

import cv2
import numpy as np


WHITE = (255, 255, 255)
GREEN = (40, 200, 40)
RED = (40, 40, 220)


def draw_text(
    frame: np.ndarray,
    text: str,
    row: int,
    color: tuple[int, int, int] = WHITE,
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


def draw_status(
    frame: np.ndarray,
    label: str,
    detected: bool,
    row: int,
) -> None:
    status = "detected" if detected else "missing"
    draw_text(frame, f"{label}: {status}", row, GREEN if detected else RED)


def draw_landmark_points(
    frame: np.ndarray,
    landmarks: list[list[float]],
    color: tuple[int, int, int],
    radius: int = 2,
) -> None:
    height, width, _ = frame.shape

    for x_norm, y_norm, _z_norm in landmarks:
        x = int(x_norm * width)
        y = int(y_norm * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), radius, color, -1)


def draw_all_landmarks(
    frame: np.ndarray,
    landmarks: dict[str, list[list[float]]],
) -> None:
    draw_landmark_points(frame, landmarks["pose"], (80, 220, 120))
    draw_landmark_points(frame, landmarks["face"], (80, 110, 255))
    draw_landmark_points(frame, landmarks["left_hand"], (255, 180, 90))
    draw_landmark_points(frame, landmarks["right_hand"], (90, 220, 255))


def compute_landmark_debug(
    landmarks: dict[str, list[list[float]]],
    features: list[float],
) -> dict:
    pose_detected = len(landmarks["pose"]) > 0
    face_detected = len(landmarks["face"]) > 0
    left_hand_detected = len(landmarks["left_hand"]) > 0
    right_hand_detected = len(landmarks["right_hand"]) > 0
    has_hand = left_hand_detected or right_hand_detected
    zero_ratio = float(np.mean(np.asarray(features, dtype=np.float32) == 0))
    frame_is_usable = pose_detected and has_hand and zero_ratio < 0.90

    return {
        "pose_detected": pose_detected,
        "face_detected": face_detected,
        "left_hand_detected": left_hand_detected,
        "right_hand_detected": right_hand_detected,
        "has_hand": has_hand,
        "zero_ratio": zero_ratio,
        "frame_is_usable": frame_is_usable,
    }


def draw_landmark_debug_overlay(
    frame: np.ndarray,
    landmarks: dict[str, list[list[float]]],
    zero_ratio: float,
    frame_is_usable: bool | None = None,
    start_row: int = 0,
) -> int:
    draw_status(frame, "Pose", len(landmarks["pose"]) > 0, start_row)
    draw_status(frame, "Face", len(landmarks["face"]) > 0, start_row + 1)
    draw_status(frame, "Left hand", len(landmarks["left_hand"]) > 0, start_row + 2)
    draw_status(frame, "Right hand", len(landmarks["right_hand"]) > 0, start_row + 3)
    draw_text(
        frame,
        f"zero ratio: {zero_ratio:.2f}",
        start_row + 4,
        GREEN if zero_ratio < 0.90 else RED,
    )

    next_row = start_row + 5
    if frame_is_usable is not None:
        draw_text(
            frame,
            f"frame usable: {'yes' if frame_is_usable else 'no'}",
            next_row,
            GREEN if frame_is_usable else RED,
        )
        next_row += 1

    return next_row
