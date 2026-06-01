# Appendix: Terminology

Short glossary for terms that came up while studying backpropagation, matrix
implementations, and the PyTorch version.

## Neural Network Structure

**Neural network**: A function made from layers of trainable parameters.

**Layer**: One network step, usually affine transform plus activation.

**Input layer**: The input values. For MNIST, `784` image pixels.

**Hidden layer**: A middle layer between input and output.

**Output layer**: The final layer. For MNIST, `10` digit scores.

**MLP**: Multi-layer perceptron; a fully connected feed-forward network.

**Feed-forward**: Data moves from input to output without loops.

**Sequential**: A simple `step -> step -> step` pipeline.

**Computation graph**: The graph of operations used to compute the output.

## Matrices And Tensors

**Scalar**: A single number.

**Vector**: A 1D array of numbers.

**Matrix**: A 2D array of numbers.

**Tensor**: A general n-dimensional array.

**Shape**: The dimensions of a tensor, e.g. `[batch, 784]`.

**Batch**: A group of examples processed together.

**Mini-batch**: A smaller batch used for one training step.

**m**: Common math symbol for number of examples in a batch.

**Transpose**: Matrix rows and columns flipped, e.g. `a2.T`.

## Linear Algebra In The Network

**Linear transformation**: Matrix multiplication, `y = W x`.

**Affine transformation**: Matrix multiplication plus bias, `y = W x + b`.

**Weights**: Trainable matrix values.

**Biases**: Trainable values added after matrix multiplication.

**Activation**: Output after applying an activation function.

**Pre-activation**: Value before activation, usually `z = W x + b`.

**Logits**: Final raw output scores before softmax/probabilities.

## Activation Functions

**Activation function**: Non-linear function inserted between affine layers.

**Sigmoid**: Squashes values to `0..1`.

**Sigmoid prime**: Derivative of sigmoid; `a * (1 - a)`.

**Prime**: Math notation for derivative.

**ReLU**: `max(0, x)`.

**Tanh**: Squashes values to `-1..1`.

**LeakyReLU**: ReLU with a small negative slope.

**GELU / SiLU**: Smooth modern activation functions.

## Cost, Loss, And Accuracy

**Cost / loss**: A number measuring how wrong the network is.

**Mean squared error**: Loss based on squared prediction error.

**Cross entropy loss**: Classification loss: `-log(probability of correct class)`.

**Softmax**: Converts scores to relative probabilities: `exp(score) / sum(exp(scores))`.

**Sigmoid vs softmax**: Sigmoid treats outputs independently; softmax makes
classes compete.

**Accuracy**: `correct / total`.

**Confusion matrix**: Table of expected class vs predicted class.

**True positive**: Correctly predicted positive case.

**False positive**: Predicted positive, but should not have.

**False negative**: Predicted negative, but should have been positive.

## Labels And Outputs

**Label**: The correct answer for an example.

**One-hot**: Class vector with one `1` and the rest `0`.

**Prediction**: The class selected by the model.

**argmax**: Index of the largest value.

## Backpropagation

**Backpropagation**: Algorithm for computing parameter gradients.

**Gradient**: Derivative of cost with respect to something.

**Gradient descent**: Updating parameters opposite the gradient.

**Learning rate**: Size of each gradient descent step.

**nabla**: Math notation often used for gradients.

**delta**: Usually `dC/dz` in backpropagation.

**dC/dw**: Derivative of cost with respect to weights.

**dC/db**: Derivative of cost with respect to biases.

**dC/dz**: Derivative of cost with respect to pre-activation.

**Chain rule**: Rule for multiplying derivatives through composed functions.

## PyTorch Terms

**Module**: Base class for PyTorch model parts.

**forward**: Method defining how inputs become outputs.

**nn.Linear**: Affine layer, `y = x @ W.T + b`.

**nn.Sigmoid**: PyTorch sigmoid activation module.

**nn.Sequential**: Container that runs modules in order.

**loss.backward()**: Computes gradients.

**optimizer.step()**: Updates parameters using gradients.

**SGD**: Stochastic gradient descent optimizer.

**DataLoader**: Creates batches from a dataset.

**TensorDataset**: Pairs tensors together as a dataset.

**dtype**: Tensor value type, e.g. `float32` or `int64`.

**device**: Where a tensor lives, usually CPU or GPU.

**autograd**: PyTorch automatic differentiation system.

## Typing Terms

**Type annotation**: Python syntax for documenting/checking types.

**Shape annotation**: Type annotation that also documents tensor dimensions.

**Pyright**: Static type checker used by the editor/LSP.

**Pyrefly**: Static type checker with tensor shape support.

**jaxtyping**: Library for annotating tensor dtype and shape.

**LSP**: Editor service for hovers, warnings, definitions, etc.

**cast**: Type-checker hint; does not change the runtime value.

## Later Study

**Conv2d**: Convolution layer, commonly used for images.

**BatchNorm1d**: Normalizes activations to help training stability.

**Dropout**: Randomly disables activations during training.

**Training mode**: Mode used while learning parameters.

**Inference mode**: Mode used while evaluating/running the trained model.
