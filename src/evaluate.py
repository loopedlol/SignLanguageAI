from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_DATA_DIR, LandmarkSequenceDataset
from model import TemporalCNN
from train import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"


@dataclass
class Prediction:
    sample_path: Path
    actual_index: int
    predicted_index: int
    confidence: float

    @property
    def is_correct(self) -> bool:
        return self.actual_index == self.predicted_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained KSL Temporal CNN.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing normalized label subfolders.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a saved training checkpoint.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Fixed number of frames per sample. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--show-correct",
        action="store_true",
        help="Also print correctly classified samples.",
    )
    return parser.parse_args()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _normalize_index_to_label(index_to_label: dict) -> dict[int, str]:
    return {int(index): label for index, label in index_to_label.items()}


def _labels_from_mapping(index_to_label: dict[int, str]) -> list[str]:
    return [index_to_label[index] for index in sorted(index_to_label)]


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict | None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"Checkpoint does not exist: {_display_path(checkpoint_path)}")
        return None

    return torch.load(checkpoint_path, map_location=device)


def warn_if_label_mismatch(
    dataset: LandmarkSequenceDataset,
    checkpoint_label_to_index: dict[str, int],
) -> None:
    if dataset.label_to_index == checkpoint_label_to_index:
        return

    print("Warning: dataset labels do not match checkpoint labels.")
    print(f"Dataset labels: {dataset.label_to_index}")
    print(f"Checkpoint labels: {checkpoint_label_to_index}")
    print()


def evaluate(
    model: TemporalCNN,
    dataset: LandmarkSequenceDataset,
    device: torch.device,
    batch_size: int = 32,
) -> list[Prediction]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[Prediction] = []
    sample_offset = 0

    model.eval()
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            logits = model(sequences)
            probabilities = torch.softmax(logits, dim=1).cpu()
            predicted_indices = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values

            for batch_index, actual_label in enumerate(labels):
                sample_path, _label_index = dataset.samples[sample_offset + batch_index]
                predictions.append(
                    Prediction(
                        sample_path=sample_path,
                        actual_index=int(actual_label.item()),
                        predicted_index=int(predicted_indices[batch_index].item()),
                        confidence=float(confidences[batch_index].item()),
                    )
                )

            sample_offset += labels.shape[0]

    return predictions


def print_per_class_accuracy(
    predictions: list[Prediction],
    labels: list[str],
) -> None:
    print("Per-class accuracy:")
    for index, label in enumerate(labels):
        class_predictions = [
            prediction for prediction in predictions if prediction.actual_index == index
        ]
        correct = sum(prediction.is_correct for prediction in class_predictions)
        total = len(class_predictions)
        print(f"{label}: {correct}/{total} correct")
    print()


def print_prediction_section(
    title: str,
    predictions: list[Prediction],
    index_to_label: dict[int, str],
) -> None:
    print(title)
    print("-" * len(title))

    if not predictions:
        print("None")
        print()
        return

    for prediction in predictions:
        actual = index_to_label.get(prediction.actual_index, str(prediction.actual_index))
        predicted = index_to_label.get(
            prediction.predicted_index,
            str(prediction.predicted_index),
        )
        print(f"sample: {_display_path(prediction.sample_path)}")
        print(f"actual: {actual}")
        print(f"predicted: {predicted}")
        print(f"confidence: {prediction.confidence:.2f}")
        print()


def print_confusion_matrix(predictions: list[Prediction], labels: list[str]) -> None:
    matrix = [
        [0 for _predicted_label in labels]
        for _actual_label in labels
    ]
    for prediction in predictions:
        if prediction.actual_index < len(labels) and prediction.predicted_index < len(labels):
            matrix[prediction.actual_index][prediction.predicted_index] += 1

    first_column_width = max(12, max((len(label) for label in labels), default=0) + 2)
    column_width = max(8, max((len(label) for label in labels), default=0) + 2)

    print("Confusion Matrix")
    print("Rows = actual, columns = predicted")
    print()
    header = " " * first_column_width + "".join(
        f"{label:>{column_width}}" for label in labels
    )
    print(header)
    for label, row in zip(labels, matrix):
        row_text = "".join(f"{count:>{column_width}}" for count in row)
        print(f"{label:<{first_column_width}}{row_text}")


def main() -> int:
    args = parse_args()
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint is None:
        return 1

    input_features = int(checkpoint["input_features"])
    num_classes = int(checkpoint["num_classes"])
    checkpoint_sequence_length = int(checkpoint["sequence_length"])
    sequence_length = args.sequence_length or checkpoint_sequence_length
    checkpoint_label_to_index = checkpoint.get("label_to_index", {})
    index_to_label = _normalize_index_to_label(checkpoint.get("index_to_label", {}))

    dataset = LandmarkSequenceDataset(
        data_dir=args.data_dir,
        sequence_length=sequence_length,
        feature_count=input_features,
    )
    if len(dataset) == 0:
        print(f"No samples found in dataset: {_display_path(dataset.data_dir)}")
        return 1

    warn_if_label_mismatch(dataset, checkpoint_label_to_index)

    model = TemporalCNN(input_features=input_features, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    predictions = evaluate(model, dataset, device)
    total_samples = len(predictions)
    correct_samples = sum(prediction.is_correct for prediction in predictions)
    accuracy = correct_samples / total_samples if total_samples else 0.0
    labels = _labels_from_mapping(index_to_label)

    print("Evaluation Summary")
    print("------------------")
    print(f"Dataset: {_display_path(dataset.data_dir)}")
    print(f"Checkpoint: {_display_path(args.checkpoint)}")
    print(f"Device: {device}")
    print(f"Total samples: {total_samples}")
    print(f"Accuracy: {accuracy:.2%}")
    print()

    print_per_class_accuracy(predictions, labels)

    incorrect_predictions = [
        prediction for prediction in predictions if not prediction.is_correct
    ]
    print_prediction_section(
        "Incorrect Predictions",
        incorrect_predictions,
        index_to_label,
    )

    if args.show_correct:
        correct_predictions = [
            prediction for prediction in predictions if prediction.is_correct
        ]
        print_prediction_section("Correct Predictions", correct_predictions, index_to_label)

    print_confusion_matrix(predictions, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
