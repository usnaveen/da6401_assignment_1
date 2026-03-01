import numpy as np


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, W, b, grad_W, grad_b):
        grad_W += self.weight_decay * W
        grad_b += self.weight_decay * b
        W -= self.lr * grad_W
        b -= self.lr * grad_b
        return W, b


class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = None
        self.v_b = None

    def update(self, W, b, grad_W, grad_b):
        if self.v_W is None:
            self.v_W = np.zeros_like(W)
            self.v_b = np.zeros_like(b)
        grad_W += self.weight_decay * W
        grad_b += self.weight_decay * b
        self.v_W = self.beta * self.v_W - self.lr * grad_W
        self.v_b = self.beta * self.v_b - self.lr * grad_b
        W += self.v_W
        b += self.v_b
        return W, b


class NAG:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = None
        self.v_b = None

    def update(self, W, b, grad_W, grad_b):
        if self.v_W is None:
            self.v_W = np.zeros_like(W)
            self.v_b = np.zeros_like(b)
        grad_W += self.weight_decay * W
        grad_b += self.weight_decay * b
        # Nesterov lookahead (using current grad as approximation)
        lookahead_grad_W = grad_W
        lookahead_grad_b = grad_b
        self.v_W = self.beta * self.v_W - self.lr * lookahead_grad_W
        self.v_b = self.beta * self.v_b - self.lr * lookahead_grad_b
        W += self.beta * self.v_W - self.lr * grad_W
        b += self.beta * self.v_b - self.lr * grad_b
        return W, b


class RMSProp:
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.rms_W = None
        self.rms_b = None

    def update(self, W, b, grad_W, grad_b):
        if self.rms_W is None:
            self.rms_W = np.zeros_like(W)
            self.rms_b = np.zeros_like(b)
        grad_W += self.weight_decay * W
        grad_b += self.weight_decay * b
        self.rms_W = self.beta * self.rms_W + (1 - self.beta) * (grad_W**2)
        self.rms_b = self.beta * self.rms_b + (1 - self.beta) * (grad_b**2)
        W -= self.lr * grad_W / (np.sqrt(self.rms_W) + self.epsilon)
        b -= self.lr * grad_b / (np.sqrt(self.rms_b) + self.epsilon)
        return W, b

