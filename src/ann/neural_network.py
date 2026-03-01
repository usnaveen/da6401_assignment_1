import numpy as np
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.objective_functions import MSE, CrossEntropy
from ann.optimizers import SGD, Momentum, NAG, RMSProp
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist
import json


class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []  # list of Dense
        self.activations = []  # list of activation classes
        self.loss = None
        self.optimizer = None
        input_size = 784
        hidden_sizes = list(
            map(int, self.args.hidden_size.split(","))
        )  # e.g., [128, 64]
        num_layers = self.args.num_layers
        assert num_layers == len(hidden_sizes) + 1, (
            "Number of layers should be number of hidden sizes + 1 (output layer)"
        )

        for i in range(num_layers - 1):
            out_size = hidden_sizes[i]
            layer = Dense(
                in_features=input_size,
                out_features=out_size,
                weight_init=self.args.weight_init,
            )
            self.layers.append(layer)
            if self.args.activation == "relu":
                act = ReLU()
            elif self.args.activation == "sigmoid":
                act = Sigmoid()
            elif self.args.activation == "tanh":
                act = Tanh()
            self.activations.append(act)
            input_size = out_size
        # Output layer (no activation for CE)
        self.layers.append(
            Dense(
                in_features=input_size,
                out_features=10,
                weight_init=self.args.weight_init,
            )
        )
        self.activations.append(None)
        # Loss
        if self.args.loss == "cross_entropy":
            self.loss = CrossEntropy()
        elif self.args.loss == "mse":
            self.loss = MSE()
        # Optimizer
        if self.args.optimizer == "sgd":
            self.optimizer = SGD(
                lr=self.args.learning_rate, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "momentum":
            self.optimizer = Momentum(
                lr=self.args.learning_rate,
                beta=0.9,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "nag":
            self.optimizer = NAG(
                lr=self.args.learning_rate,
                beta=0.9,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "rmsprop":
            self.optimizer = RMSProp(
                lr=self.args.learning_rate,
                beta=0.99,
                epsilon=1e-8,
                weight_decay=self.args.weight_decay,
            )

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns output logits
        """
        out = X
        for layer, act in zip(self.layers, self.activations):
            out = layer.forward(out)
            if act is not None:
                out = act.forward(out)
        return out

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns grad_w, grad_b (for all layers)
        """
        # Loss backward
        d_out = self.loss.backward()
        # Reverse loop over layers
        grads = []
        for layer, act in zip(reversed(self.layers), reversed(self.activations)):
            if act is not None:
                d_out = act.backward(d_out)
            d_out = layer.backward(d_out)
            grads.append((layer.grad_W, layer.grad_b))
        reversed_grads = list(reversed(grads))
        grad_w = [g[0] for g in reversed_grads]
        grad_b = [g[1] for g in reversed_grads]
        return grad_w, grad_b

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            layer.W, layer.b = self.optimizer.update(
                layer.W, layer.b, layer.grad_W, layer.grad_b
            )

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        # W&B init
        wandb.init(project=self.args.wandb_project, config=vars(self.args))
        best_f1 = 0
        for epoch in range(epochs):
            # Mini-batch loop
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]
                y_pred = self.forward(batch_X)
                loss = self.loss.forward(y_pred, batch_y)
                self.backward(batch_y, y_pred)
                self.update_weights()
            # Val metrics
            val_pred = self.forward(X_val)
            val_loss = self.loss.forward(val_pred, y_val)
            val_acc = accuracy_score(y_val, np.argmax(val_pred, axis=1))
            val_prec = precision_score(
                y_val, np.argmax(val_pred, axis=1), average="macro"
            )
            val_rec = recall_score(y_val, np.argmax(val_pred, axis=1), average="macro")
            val_f1 = f1_score(y_val, np.argmax(val_pred, axis=1), average="macro")
            # Log to W&B
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                }
            )
            # Save best on val F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                np.save("src/best_model.npy", self.get_weights())
                with open("src/best_config.json", "w") as f:
                    json.dump(vars(self.args), f)

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        pred = self.forward(X)
        acc = accuracy_score(y, np.argmax(pred, axis=1))
        prec = precision_score(
            y, np.argmax(pred, axis=1), average="macro"
        )  # fill precision (sklearn, macro)
        rec = recall_score(y, np.argmax(pred, axis=1), average="macro")
        f1 = f1_score(y, np.argmax(pred, axis=1), average="macro")
        print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")
        return acc, prec, rec, f1

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"W_{i}"] = layer.W
            weights[f"b_{i}"] = layer.b
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"W_{i}"]
            layer.b = weights[f"b_{i}"]
