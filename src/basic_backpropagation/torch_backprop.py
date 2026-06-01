"""The same MNIST pattern using modern PyTorch autograd.

This keeps the fixed architecture from `matrix_backprop.py`:

    784 -> 16 -> 16 -> 10

Here we still do forward pass, cost, backward pass, and gradient descent,
but PyTorch computes dC/dw and dC/db for us when `loss.backward()` runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import torch
from jaxtyping import Float, Int64
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from basic_backpropagation.metrics import accuracy
from basic_backpropagation.networks import MODEL_SPECS, ModelFactory
from basic_backpropagation.plots import (
    VALIDATION_COST_GRAPH_PATH,
    save_validation_cost_graph,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
RawMnistData = tuple[tuple[object, object], tuple[object, object], object]


@dataclass(frozen=True)
class MnistData:
    x_train: Float[Tensor, "train_examples 784"]
    y_train: Int64[Tensor, "train_examples"]
    x_validation: Float[Tensor, "validation_examples 784"]
    y_validation: Int64[Tensor, "validation_examples"]


@dataclass(frozen=True)
class ValidationPoint:
    iteration: int
    cost: float


@dataclass(frozen=True)
class ComparisonResult:
    model_name: str
    first_target_iteration: int | None
    validation_cost: float
    validation_accuracy: float
    validation_points: list[ValidationPoint]

MINI_BATCH_SIZE = 32
MAX_ITERATIONS = 10_000
TARGET_COST = 0.03
LEARNING_RATE = 0.5
VALIDATION_INTERVAL = 500

DATA_PATH = Path("neural-networks-and-deep-learning/data/mnist.pkl.gz")


def load_mnist_data() -> MnistData:
    with gzip.open(DATA_PATH, "rb") as f:
        raw_data: object = pickle.load(f, encoding="latin1")

    training_data, validation_data, _test_data = cast(RawMnistData, raw_data)
    train_images, train_labels = training_data
    validation_images, validation_labels = validation_data

    x_train = np.asarray(train_images, dtype=np.float64)
    y_train = np.asarray(train_labels, dtype=np.int_)
    x_validation = np.asarray(validation_images, dtype=np.float64)
    y_validation = np.asarray(validation_labels, dtype=np.int_)

    # PyTorch convention: rows are examples, columns are features.
    return MnistData(
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        x_validation=torch.tensor(x_validation, dtype=torch.float32),
        y_validation=torch.tensor(y_validation, dtype=torch.long),
    )


def validation_cost(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_function: nn.CrossEntropyLoss,
) -> float:
    total_loss = 0.0
    total_examples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = cast(tuple[Tensor, Tensor], batch)
            logits = cast(Tensor, model(x))
            loss = cast(Tensor, loss_function(logits, y))
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

    model.train()
    return total_loss / total_examples


def validation_accuracy(
    model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]]
) -> float:
    total_accuracy = 0.0
    total_examples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = cast(tuple[Tensor, Tensor], batch)
            logits = cast(Tensor, model(x))
            batch_size = x.shape[0]
            total_accuracy += accuracy(logits, y) * batch_size
            total_examples += batch_size

    model.train()
    return total_accuracy / total_examples


def train_model(
    name: str,
    create_model: ModelFactory,
    x_train: Float[Tensor, "train_examples 784"],
    y_train: Int64[Tensor, "train_examples"],
    validation_loader: DataLoader[tuple[Tensor, Tensor]],
) -> ComparisonResult:
    _ = torch.manual_seed(12345)
    batch_generator = torch.Generator().manual_seed(12345)

    training_set = TensorDataset(x_train, y_train)
    loader = DataLoader(
        training_set,
        batch_size=MINI_BATCH_SIZE,
        shuffle=True,
        generator=batch_generator,
    )

    model = create_model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    iteration: int = 0
    current_cost: float = float("inf")
    first_target_iteration: int | None = None
    validation_points = [
        ValidationPoint(0, validation_cost(model, validation_loader, loss_function))
    ]

    while iteration < MAX_ITERATIONS:
        for batch in loader:
            x, y = cast(tuple[Tensor, Tensor], batch)
            iteration += 1

            # Forward pass.
            logits = cast(Tensor, model(x))
            loss = cast(Tensor, loss_function(logits, y))
            current_cost = float(loss.item())

            # Backward pass and gradient descent.
            optimizer.zero_grad()
            loss.backward()
            _ = optimizer.step()

            if first_target_iteration is None and current_cost < TARGET_COST:
                first_target_iteration = iteration

            if iteration % VALIDATION_INTERVAL == 0:
                current_validation_cost = validation_cost(
                    model, validation_loader, loss_function
                )
                validation_points.append(
                    ValidationPoint(iteration, current_validation_cost)
                )

            if iteration % 1_000 == 0:
                print(f"{name}: iteration={iteration} cost={current_cost:.6f}")

            if iteration >= MAX_ITERATIONS:
                break

    held_out_cost = validation_cost(model, validation_loader, loss_function)
    held_out_accuracy = validation_accuracy(model, validation_loader)
    if validation_points[-1].iteration != MAX_ITERATIONS:
        validation_points.append(ValidationPoint(MAX_ITERATIONS, held_out_cost))
    return ComparisonResult(
        model_name=name,
        first_target_iteration=first_target_iteration,
        validation_cost=held_out_cost,
        validation_accuracy=held_out_accuracy,
        validation_points=validation_points,
    )


def main() -> None:
    data = load_mnist_data()
    validation_set = TensorDataset(data.x_validation, data.y_validation)
    validation_loader = cast(
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader(validation_set, batch_size=256),
    )

    results = [
        train_model(
            model_spec.name,
            model_spec.create,
            data.x_train,
            data.y_train,
            validation_loader,
        )
        for model_spec in MODEL_SPECS
    ]
    save_validation_cost_graph(
        {
            result.model_name: [
                (point.iteration, point.cost) for point in result.validation_points
            ]
            for result in results
        }
    )

    print()
    print("model, first iteration under 0.03, validation cost after 10000, validation accuracy")
    for result in results:
        target_text = (
            str(result.first_target_iteration)
            if result.first_target_iteration is not None
            else "not reached"
        )
        print(
            f"{result.model_name}, {target_text}, "
            + f"{result.validation_cost:.6f}, {result.validation_accuracy:.2%}"
        )
    print()
    print(f"Saved validation cost graph to {VALIDATION_COST_GRAPH_PATH}")


if __name__ == "__main__":
    main()
