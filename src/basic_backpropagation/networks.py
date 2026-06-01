from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

from jaxtyping import Float
from torch import Tensor, nn
from typing_extensions import override


INPUT_SIZE = 784
HIDDEN_1_SIZE = 16
WIDE_HIDDEN_1_SIZE = 32
HIDDEN_2_SIZE = 16
OUTPUT_SIZE = 10
ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    create: ModelFactory


class DigitNetwork(nn.Module):
    """Classic sequential MLP: input -> hidden1 -> hidden2 -> output."""

    layers: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_1_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_2_SIZE, OUTPUT_SIZE),
        )

    @override
    def forward(
        self, x: Float[Tensor, "batch 784"]
    ) -> Float[Tensor, "batch 10"]:
        return cast(Tensor, self.layers(x))


class InputSkipDigitNetwork(nn.Module):
    """MLP where the second hidden layer also receives the original input."""

    input_to_hidden1: nn.Linear
    hidden1_to_hidden2: nn.Linear
    input_to_hidden2: nn.Linear
    hidden2_to_output: nn.Linear
    activation: nn.Sigmoid

    def __init__(self) -> None:
        super().__init__()
        self.input_to_hidden1 = nn.Linear(INPUT_SIZE, HIDDEN_1_SIZE)
        self.hidden1_to_hidden2 = nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE)

        # The input has 784 values, while hidden2 has 16. This projection makes
        # the skip path compatible with hidden2 before adding the two paths.
        self.input_to_hidden2 = nn.Linear(INPUT_SIZE, HIDDEN_2_SIZE, bias=False)

        self.hidden2_to_output = nn.Linear(HIDDEN_2_SIZE, OUTPUT_SIZE)
        self.activation = nn.Sigmoid()

    @override
    def forward(
        self, x: Float[Tensor, "batch 784"]
    ) -> Float[Tensor, "batch 10"]:
        hidden1 = cast(Tensor, self.activation(self.input_to_hidden1(x)))

        hidden2_from_hidden1 = cast(Tensor, self.hidden1_to_hidden2(hidden1))
        hidden2_from_input = cast(Tensor, self.input_to_hidden2(x))
        hidden2 = cast(
            Tensor, self.activation(hidden2_from_hidden1 + hidden2_from_input)
        )

        return cast(Tensor, self.hidden2_to_output(hidden2))


class WideFirstDigitNetwork(nn.Module):
    """Sequential MLP with a wider first hidden layer: 784 -> 32 -> 16 -> 10."""

    layers: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, WIDE_HIDDEN_1_SIZE),
            nn.Sigmoid(),
            nn.Linear(WIDE_HIDDEN_1_SIZE, HIDDEN_2_SIZE),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_2_SIZE, OUTPUT_SIZE),
        )

    @override
    def forward(
        self, x: Float[Tensor, "batch 784"]
    ) -> Float[Tensor, "batch 10"]:
        return cast(Tensor, self.layers(x))


MODEL_SPECS = [
    ModelSpec("sequential 16->16", DigitNetwork),
    ModelSpec("input skip/projection", InputSkipDigitNetwork),
    ModelSpec("sequential 32->16", WideFirstDigitNetwork),
]
