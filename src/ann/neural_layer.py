"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np


class Dense:
    """
    Fully connected (Dense) layer.
    This is the core building block used by NeuralNetwork.
    """

    def __init__(
        self, in_features: int, out_features: int, weight_init: str = "random"
    ):
        """
        in_features:  e.g. 784 for first hidden layer
        out_features: e.g. 128 or 10 for output layer
        weight_init:  "random" or "xavier"
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init

        # ====================== WEIGHT INITIALIZATION ======================
        if weight_init == "random":
            self.W = np.random.randn(out_features, in_features) * 0.01
        elif weight_init == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, size=(out_features, in_features))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        self.b = np.zeros((1, out_features))

        # ====================== GRADIENTS (autograder needs these exact names) ======================
        self.grad_W = None
        self.grad_b = None

        # Cache for backward pass
        self.cache = None  # will store input x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (batch_size, in_features)"""
        self.cache = x.copy()  # save for backward
        out = x @ self.W.T + self.b  # matrix multiplication
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: gradient coming from next layer, shape (batch_size, out_features)

        MUST:
          - compute and STORE self.grad_W and self.grad_b
          - return dx (gradient w.r.t. this layer's input)
        """
        batch_size = dout.shape[0]

        # 1. Gradient w.r.t. weights
        self.grad_W = (self.cache.T @ dout).T  # shape must match self.W

        # 2. Gradient w.r.t. bias
        self.grad_b = np.sum(dout, axis=0, keepdims=True)  # shape (1, out_features)

        # 3. Gradient w.r.t. input (for previous layer)
        dx = dout @ self.W  # shape (batch_size, in_features)

        return dx

