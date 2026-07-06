from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import (
    INPUT_FEATURES,
    NORMALIZED_LANDMARKS_DIR,
    PROCESSED_LANDMARKS_DIR,
    PROJECT_ROOT,
)


POSE_LANDMARKS = 33
FACE_LANDMARKS = 478
HAND_LANDMARKS = 21

POSE_SIZE = POSE_LANDMARKS * 3
FACE_SIZE = FACE_LANDMARKS * 3
LEFT_HAND_SIZE = HAND_LANDMARKS * 3
RIGHT_HAND_SIZE = HAND_LANDMARKS * 3
TOTAL_FEATURES = INPUT_FEATURES

POSE_START = 0
POSE_END = POSE_START + POSE_SIZE
FACE_START = POSE_END
FACE_END = FACE_START + FACE_SIZE
LEFT_HAND_START = FACE_END
LEFT_HAND_END = LEFT_HAND_START + LEFT_HAND_SIZE
RIGHT_HAND_START = LEFT_HAND_END
RIGHT_HAND_END = RIGHT_HAND_START + RIGHT_HAND_SIZE

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
EPSILON = 1e-6

DEFAULT_INPUT_DIR = PROCESSED_LANDMARKS_DIR
DEFAULT_OUTPUT_DIR = NORMALIZED_LANDMARKS_DIR


def split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a flat 1659-value frame into pose, face, left hand, and right hand."""
    if frame.shape[0] != TOTAL_FEATURES:
        raise ValueError(f"Expected frame with {TOTAL_FEATURES} features, got {frame.shape[0]}")

    return (
        frame[POSE_START:POSE_END],
        frame[FACE_START:FACE_END],
        frame[LEFT_HAND_START:LEFT_HAND_END],
        frame[RIGHT_HAND_START:RIGHT_HAND_END],
    )


def _reshape_group(group: np.ndarray, landmark_count: int) -> np.ndarray:
    return group.reshape(landmark_count, 3)


def _normalize_group(
    group: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Normalize a landmark group, preserving all-zero missing groups."""
    if np.all(group == 0):
        return group.copy()

    return (group - origin) / scale


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize one flat landmark frame using shoulder-centered pose geometry."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (TOTAL_FEATURES,):
        raise ValueError(f"Expected frame shape ({TOTAL_FEATURES},), got {frame.shape}")

    pose_flat, face_flat, left_hand_flat, right_hand_flat = split_frame(frame)
    pose = _reshape_group(pose_flat, POSE_LANDMARKS)

    if np.all(pose == 0):
        return frame.copy()

    left_shoulder = pose[LEFT_SHOULDER_INDEX]
    right_shoulder = pose[RIGHT_SHOULDER_INDEX]

    if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
        return frame.copy()

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

    if shoulder_width < EPSILON:
        return frame.copy()

    normalized_pose = _normalize_group(pose, shoulder_center, shoulder_width)
    normalized_face = _normalize_group(
        _reshape_group(face_flat, FACE_LANDMARKS),
        shoulder_center,
        shoulder_width,
    )
    normalized_left_hand = _normalize_group(
        _reshape_group(left_hand_flat, HAND_LANDMARKS),
        shoulder_center,
        shoulder_width,
    )
    normalized_right_hand = _normalize_group(
        _reshape_group(right_hand_flat, HAND_LANDMARKS),
        shoulder_center,
        shoulder_width,
    )

    return np.concatenate(
        [
            normalized_pose.reshape(-1),
            normalized_face.reshape(-1),
            normalized_left_hand.reshape(-1),
            normalized_right_hand.reshape(-1),
        ]
    ).astype(np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize a frames x 1659 landmark sequence.

    The returned array has the same shape as the input. Frames with missing pose
    shoulders or near-zero shoulder width are copied unchanged.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {sequence.shape}")

    if sequence.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features per frame, got {sequence.shape[1]}"
        )

    normalized = np.empty_like(sequence, dtype=np.float32)
    for index, frame in enumerate(sequence):
        normalized[index] = normalize_frame(frame)

    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize recorded KSL landmark .npy sequences."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing recorded label subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where normalized label subfolders will be saved.",
    )
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_dataset(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    processed_files = 0
    skipped_files = 0

    if not input_dir.exists():
        print(f"Input directory does not exist: {_display_path(input_dir)}")
        return processed_files, skipped_files

    for sample_path in sorted(input_dir.glob("*/*.npy")):
        relative_path = sample_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        try:
            sequence = np.load(sample_path)
        except Exception as exc:
            skipped_files += 1
            print(f"Skipped {_display_path(sample_path)}: could not load file ({exc})")
            continue

        if sequence.ndim != 2:
            skipped_files += 1
            print(f"Skipped {_display_path(sample_path)}: expected 2D, got {sequence.shape}")
            continue

        if sequence.shape[1] != TOTAL_FEATURES:
            skipped_files += 1
            print(
                f"Skipped {_display_path(sample_path)}: expected {TOTAL_FEATURES} "
                f"features, got {sequence.shape[1]}"
            )
            continue

        normalized = normalize_sequence(sequence)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, normalized)
        processed_files += 1
        print(f"Saved {_display_path(output_path)} shape={normalized.shape}")

    return processed_files, skipped_files


def main() -> int:
    args = parse_args()
    processed_files, skipped_files = normalize_dataset(args.input_dir, args.output_dir)

    print()
    print("Normalization Summary")
    print("---------------------")
    print(f"Input directory: {_display_path(args.input_dir)}")
    print(f"Output directory: {_display_path(args.output_dir)}")
    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
