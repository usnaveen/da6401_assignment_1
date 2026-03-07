import numpy as np


class Dense:
    def __init__(self, in_features, out_features, weight_init="random"):
        self.in_features = in_features
        self.out_features = out_features

        if weight_init == "random":
            self.W = np.random.randn(out_features, in_features) * 0.01
        elif weight_init == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, size=(out_features, in_features))
        elif weight_init == "zero":
            self.W = np.zeros((out_features, in_features))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        self.b = np.zeros((1, out_features))
        self.grad_W = None
        self.grad_b = None
        self.cache = None

    def forward(self, x):
        self.cache = x.copy()
        return x @ self.W.T + self.b

    def backward(self, dout):
        self.grad_W = (self.cache.T @ dout).T
        self.grad_b = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W
        return dx
