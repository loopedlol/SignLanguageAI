"""Feature helpers for MediaPipe Tasks API landmark results.

``extract_landmarks`` returns variable-length landmark groups for drawing and
debugging. ``flatten_landmarks`` returns a fixed-length vector and zero-fills
missing groups so recorded samples are ready for future model training.
"""

from __future__ import annotations

from typing import Any


Landmark = list[float]
LandmarkGroup = list[Landmark]
LandmarkDict = dict[str, LandmarkGroup]

LANDMARK_GROUP_ORDER = ("pose", "face", "left_hand", "right_hand")
EXPECTED_LANDMARK_COUNTS = {
    "pose": 33,
    "face": 478,
    "left_hand": 21,
    "right_hand": 21,
}
FEATURE_COUNT = sum(EXPECTED_LANDMARK_COUNTS.values()) * 3


def _as_single_landmark_group(value: Any) -> Any:
    """Return one landmark group from common MediaPipe Tasks result shapes."""
    if value is None:
        return []

    if hasattr(value, "landmark"):
        return value.landmark

    if not isinstance(value, list) or len(value) == 0:
        return []

    first_item = value[0]
    if hasattr(first_item, "x"):
        return value

    if isinstance(first_item, list):
        return first_item

    if hasattr(first_item, "landmark"):
        return first_item.landmark

    return []


def _landmarks_to_xyz(value: Any) -> LandmarkGroup:
    """Convert MediaPipe landmarks into [[x, y, z], ...]."""
    landmark_group = _as_single_landmark_group(value)

    return [
        [float(landmark.x), float(landmark.y), float(getattr(landmark, "z", 0.0))]
        for landmark in landmark_group
        if hasattr(landmark, "x") and hasattr(landmark, "y")
    ]


def extract_landmarks(result: Any) -> LandmarkDict:
    """Extract pose, face, left hand, and right hand landmarks from a result.

    Returns:
        A dictionary where each value is a list of [x, y, z] landmarks. If a
        group is missing in the current frame, that group is returned as [].
    """
    return {
        "pose": _landmarks_to_xyz(getattr(result, "pose_landmarks", None)),
        "face": _landmarks_to_xyz(getattr(result, "face_landmarks", None)),
        "left_hand": _landmarks_to_xyz(getattr(result, "left_hand_landmarks", None)),
        "right_hand": _landmarks_to_xyz(getattr(result, "right_hand_landmarks", None)),
    }


def flatten_landmarks(landmarks: LandmarkDict) -> list[float]:
    """Flatten landmarks into a fixed-length [x, y, z, ...] feature vector.

    Missing landmarks are zero-filled. If a detected group has fewer landmarks
    than expected, the remainder is padded with zeros. If it has more, the group
    is truncated to the expected count.
    """
    flattened: list[float] = []

    for group_name in LANDMARK_GROUP_ORDER:
        expected_count = EXPECTED_LANDMARK_COUNTS[group_name]
        group = landmarks.get(group_name, [])

        for x, y, z in group[:expected_count]:
            flattened.extend([x, y, z])

        missing_count = expected_count - min(len(group), expected_count)
        flattened.extend([0.0] * missing_count * 3)

    return flattened
