from pathlib import Path

import numpy as np

try:
    from utils.simple_ml import train_test_split
except ModuleNotFoundError:
    from .simple_ml import train_test_split


SRC_DIR = Path(__file__).resolve().parents[1]


def get_artifact_path(filename):
    return SRC_DIR / filename


def _flatten_and_normalize(images):
    if images.ndim == 3:
        images = images.reshape(images.shape[0], -1)
    images = images.astype(np.float32)
    max_value = float(images.max()) if images.size else 1.0
    if max_value > 1.0:
        images /= max_value
    return images


def _load_from_npz(dataset):
    cache_names = {
        "mnist": "mnist.npz",
        "fashion_mnist": "fashion-mnist.npz",
    }
    cache_path = Path.home() / ".keras" / "datasets" / cache_names[dataset]
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        return (
            (data["x_train"], data["y_train"]),
            (data["x_test"], data["y_test"]),
        )


def _load_from_torchvision(dataset):
    try:
        from torchvision.datasets import FashionMNIST, MNIST
    except Exception:
        return None

    dataset_cls = {
        "mnist": MNIST,
        "fashion_mnist": FashionMNIST,
    }[dataset]
    root = Path.home() / ".cache" / "torchvision"

    for download in (False, True):
        try:
            train_set = dataset_cls(root=str(root), train=True, download=download)
            test_set = dataset_cls(root=str(root), train=False, download=download)
            return (
                (np.array(train_set.data), np.array(train_set.targets)),
                (np.array(test_set.data), np.array(test_set.targets)),
            )
        except Exception:
            continue

    return None


def _load_from_openml(dataset):
    dataset_names = {
        "mnist": "mnist_784",
        "fashion_mnist": "Fashion-MNIST",
    }

    try:
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml(
            dataset_names[dataset],
            version=1,
            return_X_y=True,
            as_frame=False,
        )
    except Exception:
        return None

    y = y.astype(np.int64)
    X = X.reshape(-1, 28, 28)

    if len(X) >= 70000:
        return (X[:60000], y[:60000]), (X[60000:70000], y[60000:70000])

    split = int(0.85 * len(X))
    return (X[:split], y[:split]), (X[split:], y[split:])


def _load_digits_fallback(dataset):
    try:
        from sklearn.datasets import load_digits
    except Exception:
        return _load_synthetic_fallback(dataset)

    digits = load_digits()
    images = digits.images
    labels = digits.target.astype(np.int64)

    # Upsample 8x8 digits to 28x28 so downstream code keeps the expected input size.
    upsampled = np.kron(images, np.ones((3, 3), dtype=np.float32))
    padded = np.pad(upsampled, ((0, 0), (2, 2), (2, 2)), mode="constant")

    X_train, X_test, y_train, y_test = train_test_split(
        padded,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(
        f"Warning: using sklearn digits as a fallback for {dataset}; "
        "no MNIST/Fashion-MNIST source was available."
    )
    return (X_train, y_train), (X_test, y_test)


def _load_synthetic_fallback(dataset):
    rng = np.random.RandomState(42 if dataset == "mnist" else 84)
    train_per_class = 600
    test_per_class = 100
    image_shape = (28, 28)
    prototypes = rng.uniform(0.1, 0.9, size=(10,) + image_shape).astype(np.float32)

    def _make_split(samples_per_class):
        images = []
        labels = []
        for label in range(10):
            base = np.repeat(prototypes[label][None, :, :], samples_per_class, axis=0)
            noise = rng.normal(0.0, 0.08, size=base.shape).astype(np.float32)
            images.append(np.clip(base + noise, 0.0, 1.0))
            labels.append(np.full(samples_per_class, label, dtype=np.int64))
        return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

    X_train, y_train = _make_split(train_per_class)
    X_test, y_test = _make_split(test_per_class)
    print(
        f"Warning: using synthetic fallback data for {dataset}; "
        "no dataset source was available."
    )
    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset):
    if dataset not in {"mnist", "fashion_mnist"}:
        raise ValueError(f"Unknown dataset: {dataset}")

    loaders = (
        _load_from_npz,
        _load_from_torchvision,
        _load_from_openml,
    )

    loaded = None
    for loader in loaders:
        loaded = loader(dataset)
        if loaded is not None:
            break

    if loaded is None:
        loaded = _load_digits_fallback(dataset)

    (X_train, y_train), (X_test, y_test) = loaded
    X_train = _flatten_and_normalize(np.asarray(X_train))
    X_test = _flatten_and_normalize(np.asarray(X_test))
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    return (X_train, y_train), (X_test, y_test)


class DataLoader:
    def __init__(self, dataset="mnist", batch_size=32, val_split=0.1, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.load_data()

    def load_data(self):
        (X, y), (self.X_test, self.y_test) = load_dataset(self.dataset)

        # Split train/val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.seed
        )

    def get_batch(self, X, y):
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            yield X[batch_idx], y[batch_idx]

    def get_train_batches(self):
        return self.get_batch(self.X_train, self.y_train)

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test


# Example usage (test)
if __name__ == "__main__":
    dl = DataLoader(dataset="mnist", batch_size=64)
    for batch_X, batch_y in dl.get_train_batches():
        print(batch_X.shape, batch_y.shape)
        break
    X_val, y_val = dl.get_val_data()
    print(X_val.shape, y_val.shape)
