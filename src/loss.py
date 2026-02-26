import numpy as np


class MSE:
    def __init__(self) -> None:
        self.cache = None

    def forward(self, pred, y_true):
        self.cache = pred - y_true
        loss = np.mean(self.cache**2)
        return loss

    def backward(self):
        batch_size = self.cache.shape[0]
        dx = 2 * self.cache / batch_size
        return dx


class CrossEntropy:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, y_true):
        """logits: (batch, 10), y_true: one-hot or integer labels"""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        self.probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)

        if y_true.ndim == 1:
            self.labels = np.zeros_like(self.probs)
            self.labels[np.arange(len(y_true)), y_true] = 1
        else:
            self.labels = y_true

        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        loss = -np.sum(self.labels * log_probs) / logits.shape[0]
        return loss

    def backward(self):
        """Returns dlogits, shape (batch, 10)"""
        batch_size = self.probs.shape[0]
        dlogits = (self.probs - self.labels) / batch_size
        return dlogits
