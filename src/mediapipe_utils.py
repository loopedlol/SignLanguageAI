from __future__ import annotations

import sys
from pathlib import Path


def validate_task_file(path: Path, label: str, min_size_mb: float = 1.0) -> bool:
    """Validate a MediaPipe .task file before constructing a landmarker."""
    path = path.expanduser().resolve()

    if not path.exists():
        print(
            f"\nMissing {label} MediaPipe task file.\n"
            f"Expected: {path}\n\n"
            "The file may be missing, corrupted, or overwritten. Download the "
            f"correct {label} .task model and place it at that path.\n",
            file=sys.stderr,
        )
        return False

    if path.suffix != ".task":
        print(
            f"\nInvalid {label} MediaPipe task file.\n"
            f"Expected a .task file, got: {path}\n\n"
            "The file may be missing, corrupted, or overwritten with the wrong "
            "file type.\n",
            file=sys.stderr,
        )
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= min_size_mb:
        print(
            f"\nInvalid {label} MediaPipe task file.\n"
            f"Expected: {path}\n"
            f"Size: {size_mb:.2f} MB\n\n"
            f"The file is smaller than {min_size_mb:.2f} MB and may be "
            "corrupted or overwritten. Re-download the correct .task model.\n",
            file=sys.stderr,
        )
        return False

    return True


def create_holistic_landmarker(model_path: Path):
    if not validate_task_file(model_path, "Holistic Landmarker"):
        return None
    model_path = model_path.expanduser().resolve()

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not hasattr(vision, "HolisticLandmarker") or not hasattr(
        vision,
        "HolisticLandmarkerOptions",
    ):
        print(
            "\nThis installed MediaPipe package does not expose "
            "vision.HolisticLandmarker.\n"
            "Install a MediaPipe version that includes the Holistic Landmarker "
            "Tasks API.\n",
            file=sys.stderr,
        )
        return None

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def create_hand_landmarker(model_path: Path):
    if not validate_task_file(model_path, "Hand Landmarker"):
        return None
    model_path = model_path.expanduser().resolve()

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not hasattr(vision, "HandLandmarker") or not hasattr(
        vision,
        "HandLandmarkerOptions",
    ):
        print(
            "\nThis installed MediaPipe package does not expose "
            "vision.HandLandmarker.\n",
            file=sys.stderr,
        )
        return None

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)
