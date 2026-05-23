"""Vectorized backpropagation with plain matrices.

This is a compact, Matlab-style version of the book's `network.py`.
It uses a fixed architecture:

    784 -> 16 -> 16 -> 10

Each column in X is one input example.  Each column in Y is the expected
one-hot output for that example.
"""

from __future__ import annotations

import gzip
import math
import pickle
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float64]
IntVector = NDArray[np.int_]
BoolVector = NDArray[np.bool_]
IntMatrix = NDArray[np.int_]
RawMnistData = tuple[tuple[object, object], object, object]
TrainingData = tuple[FloatMatrix, FloatMatrix]


INPUT_SIZE = 784
HIDDEN_1_SIZE = 16
HIDDEN_2_SIZE = 16
OUTPUT_SIZE = 10

MINI_BATCH_SIZE = 32
MAX_ITERATIONS = 20_000
TARGET_COST = 0.01
LEARNING_RATE = 0.5

DATA_PATH = Path("neural-networks-and-deep-learning/data/mnist.pkl.gz")


def sigmoid(z: FloatMatrix) -> FloatMatrix:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime_from_activation(a: FloatMatrix) -> FloatMatrix:
    """Return sigmoid'(z), using an already-computed activation a = sigmoid(z).
    Note: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    return a * (1.0 - a)


def one_hot(labels: IntVector) -> FloatMatrix:
    """Convert labels to one-hot columns: 7 becomes [0, 0, ..., 1, 0, 0]."""
    y = np.zeros((OUTPUT_SIZE, labels.size), dtype=np.float64)
    y[labels, np.arange(labels.size)] = 1.0
    return y


def load_mnist_training_data() -> TrainingData:
    with gzip.open(DATA_PATH, "rb") as f:
        raw_data: object = pickle.load(f, encoding="latin1")  # pyright: ignore[reportAny]

    training_data = cast(RawMnistData, raw_data)[0]
    images, labels = training_data
    images = np.asarray(images, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int_)

    # X shape: 784 x number_of_examples
    # Y shape: 10 x number_of_examples
    x = images.T
    y = one_hot(labels)
    return x, y


def cost(a3: FloatMatrix, y: FloatMatrix) -> float:
    """Mean cost of activation a3."""
    assert a3.shape == y.shape, f"expected matching shapes, got {a3.shape} and {y.shape}"
    assert a3.ndim == 2, f"expected output matrices, got {a3.ndim} dimensions"
    m = y.shape[1]
    squared_error = (a3 - y) ** 2
    return float(0.5 * np.sum(squared_error) / m)


def accuracy(a3: FloatMatrix, y: FloatMatrix) -> float:
    predictions = cast(IntVector, np.argmax(a3, axis=0))
    expected = cast(IntVector, np.argmax(y, axis=0))
    matches = cast(BoolVector, predictions == expected)
    return float(np.mean(matches))


def confusion_matrix(a3: FloatMatrix, y: FloatMatrix) -> IntMatrix:
    """Return a 10x10 matrix: rows are expected digits, columns are predicted."""
    predictions = cast(IntVector, np.argmax(a3, axis=0))
    expected = cast(IntVector, np.argmax(y, axis=0))
    matrix = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.int_)
    np.add.at(matrix, (expected, predictions), 1)
    return matrix


rng = np.random.default_rng(seed=12345)

# One weight matrix and one bias vector per layer.
# The standard deviation is scaled by 1 / sqrt(number_of_inputs) so the
# initial weighted sums do not immediately saturate the sigmoid activations.
w1: FloatMatrix = rng.normal(
    0.0, 1.0 / math.sqrt(INPUT_SIZE), (HIDDEN_1_SIZE, INPUT_SIZE)
)
b1: FloatMatrix = np.zeros((HIDDEN_1_SIZE, 1), dtype=np.float64)

w2: FloatMatrix = rng.normal(
    0.0, 1.0 / math.sqrt(HIDDEN_1_SIZE), (HIDDEN_2_SIZE, HIDDEN_1_SIZE)
)
b2: FloatMatrix = np.zeros((HIDDEN_2_SIZE, 1), dtype=np.float64)

w3: FloatMatrix = rng.normal(
    0.0, 1.0 / math.sqrt(HIDDEN_2_SIZE), (OUTPUT_SIZE, HIDDEN_2_SIZE)
)
b3: FloatMatrix = np.zeros((OUTPUT_SIZE, 1), dtype=np.float64)

x_train, y_train = load_mnist_training_data()
num_examples = x_train.shape[1]
current_cost = float("inf")

for iteration in range(1, MAX_ITERATIONS + 1):
    batch_indexes = rng.choice(num_examples, MINI_BATCH_SIZE, replace=False)
    x = x_train[:, batch_indexes]
    y = y_train[:, batch_indexes]
    # m = batchsize
    m = x.shape[1]

    # Forward pass.
    z1 = w1 @ x + b1
    a1 = sigmoid(z1)

    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)

    z3 = w3 @ a2 + b3
    a3 = sigmoid(z3)

    current_cost = cost(a3, y)
    if current_cost < TARGET_COST:
        print(f"Stopped at iteration {iteration}: cost={current_cost:.6f}")
        break



    # Layer 2 - > Output (w3)
    # z3 = w3 @ a2 + b3
    # a3 = sigmoid(z3)
    # Note: When differentiating with respect to w3, a2 is treated as the constant multiplier.
    # a3 = sigmoid(z3)
    # C = 1/2 * (a3 - y)^2

    # dC/dw3 =          dC/dz3             * dz3/dw3
    # dC/dw3 = dC/da3     * da3/dz3        * dz3/dw3
    # dC/dz2 = [(a3 - y)] * [sigmoid'(z3)]
    # dC/dw3 = [(a3 - y)] * [sigmoid'(z3)] * [a2.T]

    # Layer 1 -> Layer 2
    # z2 = w2 @ a1 + b2
    # a2 = sigmoid(z2)
    # z3 = w3 @ a2 + b3
    # dC/dw2 =              dC/dz2             * dz2/dw2
    # dC/dw2 =       dC/da2     * da2/dz2      * dz2/dw2
    # dC/dw2 = dC/dz3 * dz3/da2 * da2/dz2      * dz2/dw2
    # dC/dw2 = dC/dz3 *  w3.T   * sigmoid'(z2) * [a1.T]
    # dC/dz2 =  w3.T  * dC/dz3  * sigmoid'(z2)
    # dC/dw2 =  w3.T  * dC/dz3  * sigmoid'(z2) * [a1.T]

    # Similarly Input -> Layer 1
    # dC/dz1 =  w2.T * dC/dz2 * sigmoid'(z1)
    # dC/dw1 =  w2.T * dC/dz2 * sigmoid'(z1) * [in.T]

    # Backward pass.
    # Shapes:
    #   delta3: 10 x m
    #   delta2: 16 x m
    #   delta1: 16 x m
    # delta3 = dC/dz3 = dC/da3 * da3/dz3 = (a3 - y) * sigmoid'(z3)
    delta3 = (a3 - y) * sigmoid_prime_from_activation(a3)
    # delta2 = dC/dz2 = dC/dz3 * dz3/da2 * da2/dz2 = "w3.T * delta3 * sigmoid'(z2)"
    delta2 = (w3.T @ delta3) * sigmoid_prime_from_activation(a2)
    # delta1 = dC/dz1 = dC/dz2 * dz2/da1 * da1/dz1 = "w2.T * delta2 * sigmoid'(z1)"
    delta1 = (w2.T @ delta2) * sigmoid_prime_from_activation(a1)

    # Gradients averaged over the mini-batch.
    dw3 = delta3 @ a2.T / m
    db3 = cast(FloatMatrix, np.sum(delta3, axis=1, keepdims=True) / m)

    dw2 = delta2 @ a1.T / m
    db2 = cast(FloatMatrix, np.sum(delta2, axis=1, keepdims=True) / m)

    dw1 = delta1 @ x.T / m
    db1 = cast(FloatMatrix, np.sum(delta1, axis=1, keepdims=True) / m)

    # Gradient descent update.
    w3 -= LEARNING_RATE * dw3
    b3 -= LEARNING_RATE * db3

    w2 -= LEARNING_RATE * dw2
    b2 -= LEARNING_RATE * db2

    w1 -= LEARNING_RATE * dw1
    b1 -= LEARNING_RATE * db1

    if iteration % 500 == 0:
        print(
            f"iteration={iteration} cost={current_cost:.6f} "
            + f"batch_accuracy={accuracy(a3, y):.2%}"
        )
        print(confusion_matrix(a3, y))
else:
    print(f"Reached max iterations: cost={current_cost:.6f}")
