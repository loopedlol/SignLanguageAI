from __future__ import annotations

import cv2
import numpy as np


WHITE = (255, 255, 255)
GREEN = (40, 200, 40)
RED = (40, 40, 220)
REQUIRED_LANDMARK_MODES = ("hand", "face", "pose", "any")
ZERO_RATIO_USABLE_THRESHOLD = 0.98


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
    count: int | None = None,
) -> None:
    status = "detected" if detected else "missing"
    count_text = "" if count is None else f" | count: {count}"
    draw_text(frame, f"{label}: {status}{count_text}", row, GREEN if detected else RED)


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
    required_landmarks: str = "hand",
) -> dict:
    if required_landmarks not in REQUIRED_LANDMARK_MODES:
        supported = ", ".join(REQUIRED_LANDMARK_MODES)
        raise ValueError(
            f"Unsupported required_landmarks={required_landmarks!r}. "
            f"Choose one of: {supported}."
        )

    pose_count = len(landmarks["pose"])
    face_count = len(landmarks["face"])
    left_hand_count = len(landmarks["left_hand"])
    right_hand_count = len(landmarks["right_hand"])
    pose_detected = pose_count > 0
    face_detected = face_count > 0
    left_hand_detected = left_hand_count > 0
    right_hand_detected = right_hand_count > 0
    has_hand = left_hand_detected or right_hand_detected
    feature_array = np.asarray(features, dtype=np.float32)
    zero_ratio = 1.0 if feature_array.size == 0 else float(np.mean(feature_array == 0))

    if required_landmarks == "hand":
        requirement_met = pose_detected and has_hand
    elif required_landmarks == "face":
        requirement_met = pose_detected and face_detected
    elif required_landmarks == "pose":
        requirement_met = pose_detected
    else:
        requirement_met = pose_detected or face_detected or has_hand

    frame_is_usable = requirement_met and zero_ratio < ZERO_RATIO_USABLE_THRESHOLD

    return {
        "required_landmarks": required_landmarks,
        "pose_count": pose_count,
        "face_count": face_count,
        "left_hand_count": left_hand_count,
        "right_hand_count": right_hand_count,
        "pose_detected": pose_detected,
        "face_detected": face_detected,
        "left_hand_detected": left_hand_detected,
        "right_hand_detected": right_hand_detected,
        "has_hand": has_hand,
        "requirement_met": requirement_met,
        "zero_ratio": zero_ratio,
        "zero_ratio_threshold": ZERO_RATIO_USABLE_THRESHOLD,
        "frame_is_usable": frame_is_usable,
    }


def draw_landmark_debug_overlay(
    frame: np.ndarray,
    landmarks: dict[str, list[list[float]]],
    zero_ratio: float,
    frame_is_usable: bool | None = None,
    required_landmarks: str | None = None,
    start_row: int = 0,
) -> int:
    pose_count = len(landmarks["pose"])
    face_count = len(landmarks["face"])
    left_hand_count = len(landmarks["left_hand"])
    right_hand_count = len(landmarks["right_hand"])

    draw_status(frame, "Pose", pose_count > 0, start_row, pose_count)
    draw_status(frame, "Face", face_count > 0, start_row + 1, face_count)
    draw_status(frame, "Left hand", left_hand_count > 0, start_row + 2, left_hand_count)
    draw_status(frame, "Right hand", right_hand_count > 0, start_row + 3, right_hand_count)
    draw_text(
        frame,
        f"zero ratio: {zero_ratio:.2f}",
        start_row + 4,
        GREEN if zero_ratio < ZERO_RATIO_USABLE_THRESHOLD else RED,
    )

    next_row = start_row + 5
    if required_landmarks is not None:
        draw_text(frame, f"required landmarks: {required_landmarks}", next_row)
        next_row += 1

    if frame_is_usable is not None:
        draw_text(
            frame,
            f"frame usable: {'yes' if frame_is_usable else 'no'}",
            next_row,
            GREEN if frame_is_usable else RED,
        )
        next_row += 1

    return next_row
