import numpy as np


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x.copy()
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.cache > 0).astype(float)


class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        output = 1 / (1 + np.exp(-x))
        self.cache = output.copy()
        return output

    def backward(self, dout):
        s = self.cache
        return dout * s * (1 - s)


class Tanh:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = np.tanh(x)
        return self.cache

    def backward(self, dout):
        return dout * (1 - np.square(self.cache))


class Softmax:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)
        softmax = e_x / np.sum(e_x, axis=1, keepdims=True)
        self.cache = softmax.copy()
        return softmax

    def backward(self, dout):
        s = self.cache
        return s * (dout - np.sum(dout * s, axis=1, keepdims=True))
