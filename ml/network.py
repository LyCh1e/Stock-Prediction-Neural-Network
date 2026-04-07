"""
Pure two-layer neural network implementation.

Single Responsibility: forward pass, backward pass, weight updates, and
uncertainty estimation. No data fetching, no persistence, no orchestration.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


class NeuralNetwork:
    """
    Two-layer neural network for multi-output stock price prediction.

    Architecture: Input → Hidden (ReLU) → Output (5: open, high, low, close, volume)
    """

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 30,
        learning_rate: float = 0.001,
    ) -> None:
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.learning_rate = learning_rate

        # He initialisation for ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 5) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 5))

        self.losses:            list = []
        self.prediction_errors: list = []

    # ------------------------------------------------------------------ #
    #  Forward / Backward                                                 #
    # ------------------------------------------------------------------ #

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a forward pass through the network to produce predictions.

        Multiplies the input by the first layer's weights (W1), applies ReLU to
        introduce non-linearity, then multiplies by the second layer's weights (W2)
        to produce the final output. Returns the intermediate values z1 and a1 so
        that backward() can reuse them without recomputing.
        """
        z1     = np.dot(X, self.W1) + self.b1
        a1     = self._relu(z1)
        output = np.dot(a1, self.W2) + self.b2
        return z1, a1, output

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z1: np.ndarray,
        a1: np.ndarray,
        output: np.ndarray,
    ) -> None:
        """
        Compute gradients and update all weights and biases via gradient descent.

        Uses the chain rule to propagate the MSE loss backwards from the output
        layer through to the input layer. Gradients for each weight matrix are
        computed as dot products between upstream error signals and activations.
        If the last 10 losses are highly variable (std > 50% of mean), the
        learning rate is halved for this step to stabilise training.
        """
        m = X.shape[0]

        dL_doutput = 2 * (output - y) / m
        dL_dW2     = np.dot(a1.T, dL_doutput)
        dL_db2     = np.sum(dL_doutput, axis=0, keepdims=True)

        dL_da1 = np.dot(dL_doutput, self.W2.T)
        dL_dz1 = dL_da1 * self._relu_derivative(z1)
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Adaptive learning rate — reduce if loss is highly variable
        lr = self.learning_rate
        if len(self.losses) > 10:
            recent = self.losses[-10:]
            if np.std(recent) > np.mean(recent) * 0.5:
                lr *= 0.5

        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1

    # ------------------------------------------------------------------ #
    #  Training helpers                                                   #
    # ------------------------------------------------------------------ #

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Perform one complete training iteration on a batch of data.

        Runs a forward pass to get predictions, computes MSE loss, appends it
        to the loss history, then runs a backward pass to update the weights.
        Returns the loss so the caller can track training progress.
        """
        z1, a1, output = self.forward(X)
        loss = float(np.mean((output - y) ** 2))
        self.losses.append(loss)
        self.backward(X, y, z1, a1, output)
        return loss

    def incremental_update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Update the network weights with a single new data point or small batch.

        Intended for online/adaptive learning after initial training — e.g. when
        fresh market data arrives. Runs one forward+backward pass and records the
        mean absolute error so prediction accuracy can be tracked over time.
        """
        z1, a1, output = self.forward(X_new)
        self.backward(X_new, y_new, z1, a1, output)
        error = float(np.mean(np.abs(output - y_new)))
        self.prediction_errors.append(error)

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Produce a single deterministic prediction for the given input.

        Runs a forward pass and returns only the output array, discarding the
        intermediate values. Use this when you need a point estimate and do not
        need uncertainty information.
        """
        _, _, output = self.forward(X)
        return output

    def predict_with_uncertainty(
        self, X: np.ndarray, num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using Monte Carlo sampling.

        Runs `num_samples` forward passes, each time adding a small amount of
        Gaussian noise (mean=0, std=0.02) to the output to simulate variability.
        Returns the mean and standard deviation across all samples — a higher
        std indicates the network is less confident about that prediction.
        """
        preds = []
        for _ in range(num_samples):
            _, _, output = self.forward(X)
            preds.append(output + np.random.normal(0, 0.02, output.shape))
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    # ------------------------------------------------------------------ #
    #  Activation functions                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function element-wise.

        Clamps all negative values to zero, letting positive values pass through
        unchanged. This introduces non-linearity so the network can learn
        non-linear relationships in the data.
        """
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of ReLU element-wise.

        Returns 1 where the pre-activation value was positive, 0 elsewhere.
        Used during backpropagation to gate the gradient flowing back through
        the hidden layer — neurons that were "off" (x <= 0) receive no gradient.
        """
        return (x > 0).astype(float)
