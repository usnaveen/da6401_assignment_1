import numpy as np


class MSE:
    def __init__(self):
        self.cache = None

    def forward(self, pred, y_true):
        if y_true.ndim == 1:
            one_hot = np.zeros_like(pred)
            one_hot[np.arange(len(y_true)), y_true] = 1
            y_true = one_hot
        self.cache = pred - y_true
        return np.mean(self.cache ** 2)

    def backward(self):
        return 2 * self.cache / self.cache.shape[0]


class CrossEntropy:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        self.probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)

        if y_true.ndim == 1:
            self.labels = np.zeros_like(self.probs)
            self.labels[np.arange(len(y_true)), y_true] = 1
        else:
            self.labels = y_true

        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        return -np.sum(self.labels * log_probs) / logits.shape[0]

    def backward(self):
        return (self.probs - self.labels) / self.probs.shape[0]
