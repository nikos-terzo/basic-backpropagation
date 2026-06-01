from __future__ import annotations

from jaxtyping import Float, Int64
from torch import Tensor


def accuracy(
    logits: Float[Tensor, "batch classes"], labels: Int64[Tensor, "batch"]
) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def confusion_matrix(
    logits: Float[Tensor, "batch classes"], labels: Int64[Tensor, "batch"]
) -> Int64[Tensor, "classes classes"]:
    """Return a matrix where rows are expected classes and columns are predicted."""
    predictions = logits.argmax(dim=1)
    output_size = logits.shape[1]
    matrix = labels.new_zeros((output_size, output_size))
    for expected, predicted in zip(labels, predictions, strict=True):
        matrix[expected, predicted] += 1
    return matrix
