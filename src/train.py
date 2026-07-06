from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import FEATURE_COUNT, DEFAULT_DATA_DIR, LandmarkSequenceDataset
from model import TemporalCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KSL Temporal CNN.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing normalized label subfolders.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Fixed number of frames per sample.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for latest.pt, best.pt, and label_mapping.json.",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def split_indices(
    dataset_size: int,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 < val_split < 1:
        raise ValueError("--val-split must be greater than 0 and less than 1.")

    indices = list(range(dataset_size))
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    indices = [indices[index] for index in shuffled]

    val_size = int(round(dataset_size * val_split))
    val_size = max(1, min(val_size, dataset_size - 1))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def make_loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(sequences)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def _json_safe_index_to_label(index_to_label: dict[int, str]) -> dict[str, str]:
    return {str(index): label for index, label in index_to_label.items()}


def save_label_mapping(dataset: LandmarkSequenceDataset, checkpoint_dir: Path) -> None:
    mapping_path = checkpoint_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "label_to_index": dataset.label_to_index,
                "index_to_label": _json_safe_index_to_label(dataset.index_to_label),
            },
            file,
            indent=2,
            sort_keys=True,
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    dataset: LandmarkSequenceDataset,
    sequence_length: int,
    input_features: int,
    num_classes: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "label_to_index": dataset.label_to_index,
            "index_to_label": dataset.index_to_label,
            "sequence_length": sequence_length,
            "input_features": input_features,
            "num_classes": num_classes,
        },
        path,
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()

    if args.sequence_length <= 0:
        print("--sequence-length must be greater than 0.")
        return 1
    if args.epochs <= 0:
        print("--epochs must be greater than 0.")
        return 1
    if args.batch_size <= 0:
        print("--batch-size must be greater than 0.")
        return 1
    if args.lr <= 0:
        print("--lr must be greater than 0.")
        return 1

    try:
        set_seed(args.seed)
        dataset = LandmarkSequenceDataset(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            feature_count=FEATURE_COUNT,
        )

        if len(dataset) < 2:
            print(
                "Dataset must contain at least 2 samples to create train and "
                "validation splits."
            )
            return 1

        num_classes = len(dataset.label_to_index)
        if num_classes < 2:
            print("Warning: classification training needs at least 2 labels/classes.")

        train_indices, val_indices = split_indices(len(dataset), args.val_split, args.seed)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    device = get_device()
    train_loader = make_loader(
        dataset,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        dataset,
        val_indices,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = TemporalCNN(
        input_features=FEATURE_COUNT,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_label_mapping(dataset, checkpoint_dir)

    print(f"Data directory: {_display_path(dataset.data_dir)}")
    print(f"Checkpoint directory: {_display_path(checkpoint_dir)}")
    print(f"Device: {device}")
    print(f"Labels: {dataset.label_to_index}")
    print(f"Samples: {len(dataset)} | Train: {len(train_indices)} | Val: {len(val_indices)}")
    print()

    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f}")
        print(f"Val loss: {val_loss:.4f} | Val acc: {val_accuracy:.4f}")
        print()

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            val_accuracy,
            dataset,
            args.sequence_length,
            FEATURE_COUNT,
            num_classes,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_path = checkpoint_dir / "best.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                val_accuracy,
                dataset,
                args.sequence_length,
                FEATURE_COUNT,
                num_classes,
            )

    print(f"Saved latest checkpoint: {_display_path(checkpoint_dir / 'latest.pt')}")
    print(f"Saved best checkpoint: {_display_path(checkpoint_dir / 'best.pt')}")
    print(f"Saved label mapping: {_display_path(checkpoint_dir / 'label_mapping.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
