from __future__ import annotations

import torch
from torch import nn

from config import INPUT_FEATURES, SEQUENCE_LENGTH, TRAIN_DROPOUT


DEFAULT_INPUT_FEATURES = INPUT_FEATURES
DEFAULT_NUM_CLASSES = 3
DEFAULT_DROPOUT = TRAIN_DROPOUT
DEMO_BATCH_SIZE = 8
DEMO_SEQUENCE_LENGTH = SEQUENCE_LENGTH


class TemporalCNN(nn.Module):
    """Simple Temporal CNN for isolated KSL landmark sequence classification."""

    def __init__(
        self,
        input_features: int = DEFAULT_INPUT_FEATURES,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be greater than 0.")
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.input_features = input_features
        self.num_classes = num_classes

        self.temporal_features = nn.Sequential(
            nn.Conv1d(input_features, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TemporalCNN expects input with shape "
                "batch x sequence_length x input_features."
            )

        if x.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected final dimension {self.input_features}, got {x.shape[-1]}."
            )

        x = x.transpose(1, 2)
        x = self.temporal_features(x)
        x = x.mean(dim=2)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> int:
    x = torch.randn(DEMO_BATCH_SIZE, DEMO_SEQUENCE_LENGTH, DEFAULT_INPUT_FEATURES)
    model = TemporalCNN(input_features=DEFAULT_INPUT_FEATURES, num_classes=5)
    logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameter count: {count_parameters(model):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
