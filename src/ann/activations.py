"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class ReLU:
    def __init__(self) -> None:
        self.cache = None

    def forward(self, x) -> float:
        """x shape: (batch_size, features)"""
        self.cache = x.copy()
        return np.maximum(0, x)

    def backward(self, dout):
        """dout shape same as forward"""
        dx = dout * (self.cache > 0).astype(float)
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.cache = None

    def forward(self, x):
        """x shape: (batch_size, features)"""

        output = 1 / (1 + np.exp(-x))
        self.cache = output.copy()
        return output

    def backward(self, dout):
        """dout shape same as forward"""

        x = self.cache

        dx = dout * x(1 - x)

        return dx


class Tanh:
    def __init__(self) -> None:
        self.cache = None

    def forward(self, x):
        """x shape: (batch_size, features)"""

        self.cache = np.tanh(x)
        return self.cache

    def backward(self, dout):
        """dout shape same as forward"""

        dx = dout * (1 - np.square(self.cache))
        return dx


class Softmax:
    def __init__(self) -> None:
        self.cache = None

    def forward(self, x):
        """x shape: (batch_size, num_classes)"""
        # Stability trick: subtract max per sample to prevent overflow
        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)

        # Normalize per sample (axis=1)
        softmax = e_x / np.sum(e_x, axis=1, keepdims=True)

        self.cache = softmax.copy()  # cache the probabilities
        return softmax

    def backward(self, dout):
        """dout shape same as forward output"""
        # Efficient vectorized gradient of softmax:
        # dx = S * (dout - sum(dout * S, axis=1))
        s = self.cache
        dx = s * (dout - np.sum(dout * s, axis=1, keepdims=True))
        return dx
