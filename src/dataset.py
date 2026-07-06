from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import INPUT_FEATURES, NORMALIZED_LANDMARKS_DIR, PROJECT_ROOT, SEQUENCE_LENGTH


DEFAULT_DATA_DIR = NORMALIZED_LANDMARKS_DIR
DEFAULT_SEQUENCE_LENGTH = SEQUENCE_LENGTH
FEATURE_COUNT = INPUT_FEATURES


class LandmarkSequenceDataset(Dataset):
    """PyTorch dataset for fixed-length normalized KSL landmark sequences."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        feature_count: int = FEATURE_COUNT,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0.")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0.")

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.label_to_index = self._build_label_mapping()
        self.index_to_label = {
            index: label for label, index in self.label_to_index.items()
        }
        self.samples = self._collect_samples()

    def _build_label_mapping(self) -> dict[str, int]:
        if not self.data_dir.exists():
            return {}

        labels = sorted(path.name for path in self.data_dir.iterdir() if path.is_dir())
        return {label: index for index, label in enumerate(labels)}

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        for label, label_index in self.label_to_index.items():
            label_dir = self.data_dir / label
            for sample_path in sorted(label_dir.glob("*.npy")):
                samples.append((sample_path, label_index))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_path, label_index = self.samples[index]
        sequence = np.load(sample_path)
        sequence = self._validate_sequence(sequence, sample_path)
        sequence = self._pad_or_trim(sequence)

        sequence_tensor = torch.as_tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label_index, dtype=torch.long)
        return sequence_tensor, label_tensor

    def _validate_sequence(self, sequence: np.ndarray, sample_path: Path) -> np.ndarray:
        if sequence.ndim != 2:
            raise ValueError(
                f"{sample_path} has shape {sequence.shape}; expected frames x features."
            )

        if sequence.shape[1] != self.feature_count:
            raise ValueError(
                f"{sample_path} has {sequence.shape[1]} features; "
                f"expected {self.feature_count}."
            )

        return sequence.astype(np.float32, copy=False)

    def _pad_or_trim(self, sequence: np.ndarray) -> np.ndarray:
        frame_count = sequence.shape[0]

        if frame_count >= self.sequence_length:
            return sequence[: self.sequence_length]

        padded = np.zeros(
            (self.sequence_length, self.feature_count),
            dtype=np.float32,
        )
        padded[:frame_count] = sequence
        return padded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load normalized KSL landmark sequences as a PyTorch dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing normalized label subfolders.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Fixed number of frames returned per sample.",
    )
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()

    try:
        dataset = LandmarkSequenceDataset(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Dataset directory: {_display_path(dataset.data_dir)}")
    print(f"Labels found: {len(dataset.label_to_index)}")
    print(f"Total samples: {len(dataset)}")
    print(f"Label mapping: {dataset.label_to_index}")

    if len(dataset) == 0:
        print("First sample shape: n/a")
        print("First label: n/a")
        return 0

    try:
        sequence, label = dataset[0]
    except Exception as exc:
        print(f"Could not load first sample: {exc}")
        return 1

    print(f"First sample shape: {sequence.shape}")
    print(f"First label: {label.item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
