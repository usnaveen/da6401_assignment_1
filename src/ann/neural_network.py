import json
import numpy as np

try:
    from ann.neural_layer import Dense
    from ann.activations import ReLU, Sigmoid, Tanh, Softmax
    from ann.objective_functions import MSE, CrossEntropy
    from ann.optimizers import SGD, Momentum, NAG, RMSProp
    from utils.data_loader import get_artifact_path
    from utils.simple_ml import (
        accuracy_score, f1_score, precision_score, recall_score, train_test_split,
    )
except ModuleNotFoundError:
    from .neural_layer import Dense
    from .activations import ReLU, Sigmoid, Tanh, Softmax
    from .objective_functions import MSE, CrossEntropy
    from .optimizers import SGD, Momentum, NAG, RMSProp
    from ..utils.data_loader import get_artifact_path
    from ..utils.simple_ml import (
        accuracy_score, f1_score, precision_score, recall_score, train_test_split,
    )

try:
    import wandb
except Exception:
    class _WandbStub:
        def init(self, *a, **kw): return None
        def log(self, *a, **kw): return None
        def finish(self, *a, **kw): return None
    wandb = _WandbStub()


def _get_first_attr(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def _parse_hidden_sizes(raw):
    if raw is None:
        return [128, 128]
    if isinstance(raw, str):
        cleaned = raw.strip().strip("[]()")
        if not cleaned:
            return [128, 128]
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 1 and " " in parts[0]:
            parts = [p for p in parts[0].split() if p]
        return [int(float(p)) for p in parts]
    if isinstance(raw, (list, tuple, np.ndarray)):
        parsed = []
        for item in raw:
            if isinstance(item, str) and "," in item:
                parsed.extend(int(float(p.strip())) for p in item.split(",") if p.strip())
            else:
                parsed.append(int(float(item)))
        return parsed if parsed else [128, 128]
    return [int(float(raw))]


def _safe_wandb_log(data, step=None):
    try:
        wandb.log(data) if step is None else wandb.log(data, step=step)
    except Exception:
        pass


class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []
        self.activations = []
        self.loss = None
        self.optimizer = None
        self.activation_cache = []
        input_size = 784

        raw_hidden = _get_first_attr(
            self.args,
            ["hidden_size", "hidden_sizes", "hidden_layer_size", "hidden_layer_sizes"],
            "128,128",
        )
        hidden_sizes = _parse_hidden_sizes(raw_hidden)

        num_hidden = int(_get_first_attr(
            self.args, ["num_layers", "num_hidden_layers", "hidden_layers"], len(hidden_sizes),
        ))

        if len(hidden_sizes) == 1 and num_hidden > 1:
            hidden_sizes = hidden_sizes * num_hidden
        elif num_hidden != len(hidden_sizes):
            num_hidden = len(hidden_sizes)

        self.args.hidden_size = ",".join(map(str, hidden_sizes))
        self.args.num_layers = num_hidden

        activation_name = str(_get_first_attr(self.args, ["activation", "act_fn"], "relu")).lower()
        loss_name = str(_get_first_attr(self.args, ["loss", "loss_function"], "cross_entropy")).lower()
        optimizer_name = str(_get_first_attr(self.args, ["optimizer", "optimiser"], "rmsprop")).lower()
        weight_init = str(_get_first_attr(self.args, ["weight_init", "weight_initialization"], "xavier")).lower()
        lr = float(_get_first_attr(self.args, ["learning_rate", "lr"], 0.001))
        wd = float(_get_first_attr(self.args, ["weight_decay", "wd"], 0.0))

        # Hidden layers
        for i in range(num_hidden):
            self.layers.append(Dense(input_size, hidden_sizes[i], weight_init))
            self.activations.append(self._make_activation(activation_name))
            input_size = hidden_sizes[i]

        # Output layer (logits, no activation)
        self.layers.append(Dense(input_size, 10, weight_init))
        self.activations.append(None)

        # Loss
        if loss_name in {"cross_entropy", "ce"}:
            self.loss = CrossEntropy()
        elif loss_name in {"mse", "mean_squared_error"}:
            self.loss = MSE()
        else:
            self.loss = CrossEntropy()

        # Optimizer
        if optimizer_name == "sgd":
            self.optimizer = SGD(lr=lr, weight_decay=wd)
        elif optimizer_name == "momentum":
            self.optimizer = Momentum(lr=lr, beta=0.9, weight_decay=wd)
        elif optimizer_name in {"nag", "nesterov"}:
            self.optimizer = NAG(lr=lr, beta=0.9, weight_decay=wd)
        elif optimizer_name == "rmsprop":
            self.optimizer = RMSProp(lr=lr, beta=0.99, epsilon=1e-8, weight_decay=wd)
        else:
            self.optimizer = RMSProp(lr=lr, beta=0.99, epsilon=1e-8, weight_decay=wd)

    def _make_activation(self, name):
        name = str(name).lower()
        if name == "relu": return ReLU()
        if name == "sigmoid": return Sigmoid()
        if name == "tanh": return Tanh()
        return ReLU()

    def forward(self, X, store_activations=False):
        out = X
        if store_activations:
            self.activation_cache = []
        for layer, act in zip(self.layers, self.activations):
            out = layer.forward(out)
            if act is not None:
                out = act.forward(out)
                if store_activations:
                    self.activation_cache.append(out.copy())
        self.last_logits = out
        return out

    def backward(self, y_true=None, y_pred=None):
        if y_true is not None and y_pred is not None:
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)
            is_logits = lambda a: a.ndim == 2 and a.shape[1] == 10
            if is_logits(y_true_arr) and not is_logits(y_pred_arr):
                y_true_arr, y_pred_arr = y_pred_arr, y_true_arr
            self.loss.forward(y_pred_arr, y_true_arr)

        d_out = self.loss.backward()
        grads = []
        for layer, act in zip(reversed(self.layers), reversed(self.activations)):
            if act is not None:
                d_out = act.backward(d_out)
            d_out = layer.backward(d_out)
            grads.append((layer.grad_W, layer.grad_b))
        return [g[0] for g in grads], [g[1] for g in grads]

    def update_weights(self):
        for i, layer in enumerate(self.layers):
            layer.W, layer.b = self.optimizer.update(
                layer.W, layer.b, layer.grad_W, layer.grad_b, layer_id=str(i)
            )

    def get_gradient_norms(self):
        norms = {}
        for i, layer in enumerate(self.layers):
            if layer.grad_W is not None:
                norms[f"grad_norm_layer_{i}"] = float(np.linalg.norm(layer.grad_W))
        return norms

    def get_neuron_gradients(self, layer_idx, neuron_indices):
        layer = self.layers[layer_idx]
        result = {}
        if layer.grad_W is not None:
            for ni in neuron_indices:
                if ni < layer.grad_W.shape[0]:
                    result[f"layer{layer_idx}_neuron{ni}_grad_norm"] = float(
                        np.linalg.norm(layer.grad_W[ni])
                    )
        return result

    def get_activation_stats(self):
        stats = {}
        for i, act_vals in enumerate(self.activation_cache):
            stats[f"dead_neuron_frac_layer_{i}"] = float(np.mean(np.all(act_vals == 0, axis=0)))
            stats[f"zero_activation_frac_layer_{i}"] = float(np.mean(act_vals == 0))
            stats[f"mean_activation_layer_{i}"] = float(np.mean(np.abs(act_vals)))
        return stats

    def train(self, X_train, y_train, epochs, batch_size, X_test=None, y_test=None,
              log_gradients=False, log_activations=False, log_neuron_gradients=False,
              neuron_layer_idx=0, neuron_indices=None, max_iterations=None,
              save_artifacts=True):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        if neuron_indices is None:
            neuron_indices = [0, 1, 2, 3, 4]

        best_f1 = float("-inf")
        global_step = 0
        model_path = get_artifact_path("best_model.npy")
        config_path = get_artifact_path("best_config.json")

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_shuf = X_train[indices]
            y_shuf = y_train[indices]

            for i in range(0, len(X_shuf), batch_size):
                batch_X = X_shuf[i:i + batch_size]
                batch_y = y_shuf[i:i + batch_size]

                y_pred = self.forward(batch_X, store_activations=log_activations)
                loss = self.loss.forward(y_pred, batch_y)
                self.backward(batch_y, y_pred)
                self.update_weights()
                global_step += 1

                if log_gradients or log_neuron_gradients or log_activations:
                    step_log = {"iteration": global_step, "batch_loss": float(loss)}
                    if log_gradients:
                        step_log.update(self.get_gradient_norms())
                    if log_neuron_gradients:
                        step_log.update(self.get_neuron_gradients(neuron_layer_idx, neuron_indices))
                    if log_activations:
                        step_log.update(self.get_activation_stats())
                    _safe_wandb_log(step_log, step=global_step)

                if max_iterations and global_step >= max_iterations:
                    print(f"Stopped at iteration {global_step}")
                    return

            # Epoch-level metrics
            train_pred = self.forward(X_train)
            train_loss = self.loss.forward(train_pred, y_train)
            train_acc = accuracy_score(y_train, np.argmax(train_pred, axis=1))

            val_pred = self.forward(X_val)
            val_loss = self.loss.forward(val_pred, y_val)
            val_labels = np.argmax(val_pred, axis=1)
            val_acc = accuracy_score(y_val, val_labels)
            val_prec = precision_score(y_val, val_labels, average="macro", zero_division=0)
            val_rec = recall_score(y_val, val_labels, average="macro", zero_division=0)
            val_f1 = f1_score(y_val, val_labels, average="macro", zero_division=0)

            log_dict = {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_precision": val_prec, "val_recall": val_rec, "val_f1": val_f1,
            }

            if X_test is not None and y_test is not None:
                test_pred = self.forward(X_test)
                test_loss = self.loss.forward(test_pred, y_test)
                test_labels = np.argmax(test_pred, axis=1)
                test_acc = accuracy_score(y_test, test_labels)
                test_f1 = f1_score(y_test, test_labels, average="macro", zero_division=0)
                log_dict.update({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})

                if save_artifacts and test_f1 > best_f1:
                    best_f1 = test_f1
                    np.save(model_path, self.get_weights())
                    with open(config_path, "w") as f:
                        json.dump(vars(self.args), f)
            else:
                if save_artifacts and val_f1 > best_f1:
                    best_f1 = val_f1
                    np.save(model_path, self.get_weights())
                    with open(config_path, "w") as f:
                        json.dump(vars(self.args), f)

            _safe_wandb_log(log_dict)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

    def evaluate(self, X, y):
        pred = self.forward(X)
        pred_labels = np.argmax(pred, axis=1)
        acc = accuracy_score(y, pred_labels)
        prec = precision_score(y, pred_labels, average="macro", zero_division=0)
        rec = recall_score(y, pred_labels, average="macro", zero_division=0)
        f1 = f1_score(y, pred_labels, average="macro", zero_division=0)
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        return acc, prec, rec, f1

    def get_weights(self):
        return {f"W_{i}": l.W for i, l in enumerate(self.layers)} | \
               {f"b_{i}": l.b for i, l in enumerate(self.layers)}

    def _normalize_weight_entries(self, weights):
        if isinstance(weights, (list, tuple)):
            normalized = []
            for item in weights:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise TypeError("weights items must be (W, b) pairs")
                normalized.append((np.array(item[0]), np.array(item[1])))
            return normalized

        if not isinstance(weights, dict):
            raise TypeError("weights must be a dict or list/tuple of (W, b)")

        normalized = []
        layer_idx = 0
        while True:
            w_keys = [f"W_{layer_idx}", f"W{layer_idx}", f"layer_{layer_idx}_W", f"weights_{layer_idx}", f"weight_{layer_idx}"]
            b_keys = [f"b_{layer_idx}", f"b{layer_idx}", f"layer_{layer_idx}_b", f"bias_{layer_idx}", f"biases_{layer_idx}"]
            w_key = next((k for k in w_keys if k in weights), None)
            b_key = next((k for k in b_keys if k in weights), None)

            if w_key is None or b_key is None:
                nk = str(layer_idx)
                if nk in weights and isinstance(weights[nk], dict):
                    nested = weights[nk]
                    w_key = next(((nk, k) for k in ("W", "weight", "weights") if k in nested), None)
                    b_key = next(((nk, k) for k in ("b", "bias", "biases") if k in nested), None)

            if w_key is None or b_key is None:
                break

            if isinstance(w_key, tuple):
                w_val = np.array(weights[w_key[0]][w_key[1]])
            else:
                w_val = np.array(weights[w_key])
            if isinstance(b_key, tuple):
                b_val = np.array(weights[b_key[0]][b_key[1]])
            else:
                b_val = np.array(weights[b_key])

            normalized.append((w_val, b_val))
            layer_idx += 1

        if not normalized:
            raise KeyError(f"No layer weights found. Keys: {list(weights.keys())[:20]}")
        return normalized

    def _rebuild_layers_from_weights(self, normalized_weights):
        act_name = _get_first_attr(self.args, ["activation", "act_fn"], "relu")
        w_init = _get_first_attr(self.args, ["weight_init", "weight_initialization"], "xavier")

        self.layers = []
        self.activations = []
        hidden_sizes = []

        for idx, (w_val, _) in enumerate(normalized_weights):
            w_val = np.array(w_val)
            if w_val.ndim != 2:
                raise ValueError(f"Layer {idx}: expected 2D weights, got {w_val.shape}")
            out_f, in_f = w_val.shape
            self.layers.append(Dense(in_f, out_f, w_init))
            if idx < len(normalized_weights) - 1:
                self.activations.append(self._make_activation(act_name))
                hidden_sizes.append(out_f)
            else:
                self.activations.append(None)

        self.args.hidden_size = ",".join(map(str, hidden_sizes)) if hidden_sizes else ""
        self.args.num_layers = len(hidden_sizes)

    def set_weights(self, weights):
        normalized = self._normalize_weight_entries(weights)

        def _arch_matches():
            if len(normalized) != len(self.layers):
                return False
            for layer, (w, _) in zip(self.layers, normalized):
                if w.shape == layer.W.shape:
                    continue
                if w.ndim == 2 and w.T.shape == layer.W.shape:
                    continue
                return False
            return True

        if not _arch_matches():
            self._rebuild_layers_from_weights(normalized)

        for i, (layer, (w_val, b_val)) in enumerate(zip(self.layers, normalized)):
            if w_val.shape == layer.W.shape:
                layer.W = np.array(w_val)
            elif w_val.ndim == 2 and w_val.T.shape == layer.W.shape:
                layer.W = np.array(w_val).T
            else:
                raise ValueError(f"Layer {i}: incompatible W shape {w_val.shape}, expected {layer.W.shape}")

            b_val = np.array(b_val)
            if b_val.ndim == 1:
                b_val = b_val.reshape(1, -1)
            elif b_val.ndim == 2 and b_val.shape[1] == 1:
                b_val = b_val.T
            if b_val.shape != layer.b.shape:
                raise ValueError(f"Layer {i}: incompatible b shape {b_val.shape}, expected {layer.b.shape}")
            layer.b = b_val
