import numpy as np


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, W, b, grad_W, grad_b, layer_id=None):
        gW = grad_W + self.weight_decay * W
        gb = grad_b + self.weight_decay * b
        return W - self.lr * gW, b - self.lr * gb


class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.state = {}

    def update(self, W, b, grad_W, grad_b, layer_id):
        if layer_id not in self.state:
            self.state[layer_id] = {"v_W": np.zeros_like(W), "v_b": np.zeros_like(b)}
        s = self.state[layer_id]
        gW = grad_W + self.weight_decay * W
        gb = grad_b + self.weight_decay * b
        s["v_W"] = self.beta * s["v_W"] - self.lr * gW
        s["v_b"] = self.beta * s["v_b"] - self.lr * gb
        return W + s["v_W"], b + s["v_b"]


class NAG:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.state = {}

    def update(self, W, b, grad_W, grad_b, layer_id):
        if layer_id not in self.state:
            self.state[layer_id] = {"v_W": np.zeros_like(W), "v_b": np.zeros_like(b)}
        s = self.state[layer_id]
        gW = grad_W + self.weight_decay * W
        gb = grad_b + self.weight_decay * b
        v_W_prev = s["v_W"].copy()
        v_b_prev = s["v_b"].copy()
        s["v_W"] = self.beta * s["v_W"] - self.lr * gW
        s["v_b"] = self.beta * s["v_b"] - self.lr * gb
        W = W - self.beta * v_W_prev + (1 + self.beta) * s["v_W"]
        b = b - self.beta * v_b_prev + (1 + self.beta) * s["v_b"]
        return W, b


class RMSProp:
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.state = {}

    def update(self, W, b, grad_W, grad_b, layer_id):
        if layer_id not in self.state:
            self.state[layer_id] = {"rms_W": np.zeros_like(W), "rms_b": np.zeros_like(b)}
        s = self.state[layer_id]
        gW = grad_W + self.weight_decay * W
        gb = grad_b + self.weight_decay * b
        s["rms_W"] = self.beta * s["rms_W"] + (1 - self.beta) * (gW ** 2)
        s["rms_b"] = self.beta * s["rms_b"] + (1 - self.beta) * (gb ** 2)
        W = W - self.lr * gW / (np.sqrt(s["rms_W"]) + self.epsilon)
        b = b - self.lr * gb / (np.sqrt(s["rms_b"]) + self.epsilon)
        return W, b
