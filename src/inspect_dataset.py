from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed_landmarks"
DEFAULT_EXPECTED_FEATURES = 1659
ZERO_RATIO_WARNING_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect recorded KSL landmark .npy dataset samples."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing label subfolders with .npy samples.",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=DEFAULT_EXPECTED_FEATURES,
        help="Expected feature count for each frame.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Recommended minimum sample count per label.",
    )
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _iter_label_dirs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []

    return sorted(path for path in data_dir.iterdir() if path.is_dir())


def _format_average(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def inspect_dataset(
    data_dir: Path,
    expected_features: int,
    min_samples: int,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    label_dirs = _iter_label_dirs(data_dir)
    label_counts: dict[str, int] = {}
    frame_counts: list[int] = []
    file_shapes: list[str] = []
    valid_files = 0
    invalid_files = 0
    warnings: list[str] = []

    if not data_dir.exists():
        warnings.append(f"Data directory does not exist: {_display_path(data_dir)}")

    for label_dir in label_dirs:
        sample_paths = sorted(label_dir.glob("*.npy"))
        label_counts[label_dir.name] = len(sample_paths)

        if len(sample_paths) < min_samples:
            warnings.append(
                f"{label_dir.name} has only {len(sample_paths)} samples, "
                f"below recommended minimum of {min_samples}"
            )

        for sample_path in sample_paths:
            try:
                arr = np.load(sample_path)
            except Exception as exc:
                invalid_files += 1
                file_shapes.append(f"{_display_path(sample_path)}: load error")
                warnings.append(f"{_display_path(sample_path)} could not be loaded: {exc}")
                continue

            file_shapes.append(f"{_display_path(sample_path)}: {arr.shape}")

            if arr.size == 0:
                invalid_files += 1
                warnings.append(f"{_display_path(sample_path)} is empty")
                continue

            if arr.ndim != 2:
                invalid_files += 1
                warnings.append(
                    f"{_display_path(sample_path)} has invalid shape {arr.shape}; "
                    "expected frames x features"
                )
                continue

            frames, features = arr.shape
            frame_counts.append(frames)

            if features != expected_features:
                invalid_files += 1
                warnings.append(
                    f"{_display_path(sample_path)} has {features} features; "
                    f"expected {expected_features}"
                )
                continue

            valid_files += 1
            zero_ratio = float(np.mean(arr == 0))
            if zero_ratio > ZERO_RATIO_WARNING_THRESHOLD:
                warnings.append(
                    f"{_display_path(sample_path)} is {zero_ratio:.0%} zeros"
                )

    min_frames = min(frame_counts) if frame_counts else None
    max_frames = max(frame_counts) if frame_counts else None
    avg_frames = float(np.mean(frame_counts)) if frame_counts else None

    print("Dataset Summary")
    print("---------------")
    print(f"Data directory: {_display_path(data_dir)}")
    print(f"Labels found: {len(label_dirs)}")
    print(f"Expected feature count: {expected_features}")
    print()

    print("Per-label counts:")
    if label_counts:
        for label, count in label_counts.items():
            print(f"{label}: {count} samples")
    else:
        print("No labels found")
    print()

    print("Frame length stats:")
    print(f"min frames: {min_frames if min_frames is not None else 'n/a'}")
    print(f"max frames: {max_frames if max_frames is not None else 'n/a'}")
    print(f"avg frames: {_format_average(avg_frames)}")
    print()

    print("Feature shape check:")
    print(f"valid files: {valid_files}")
    print(f"invalid files: {invalid_files}")
    print()

    print("File shapes:")
    if file_shapes:
        for file_shape in file_shapes:
            print(file_shape)
    else:
        print("No .npy files found")
    print()

    print("Warnings:")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("None")


def main() -> int:
    args = parse_args()

    if args.expected_features <= 0:
        print("--expected-features must be greater than 0.")
        return 1

    if args.min_samples < 0:
        print("--min-samples must be 0 or greater.")
        return 1

    inspect_dataset(args.data_dir, args.expected_features, args.min_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
