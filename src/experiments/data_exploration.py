"""
Section 2.1 — Data Exploration (3 marks)
Logs a W&B Table with 5 sample images per class.

Run from project root:
    cd src && python -m experiments.data_exploration
"""
import os
from pathlib import Path

import numpy as np
import wandb
from utils.data_loader import load_dataset

MNIST_CLASS_NAMES = [str(i) for i in range(10)]
FASHION_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def _configure_local_wandb_dirs():
    """
    Ensure W&B uses writable project-local dirs.
    This avoids cache/artifact permission issues on some environments.
    """
    src_root = Path(__file__).resolve().parents[1]
    paths = {
        "WANDB_DIR": src_root / "wandb",
        "WANDB_CACHE_DIR": src_root / ".wandb_cache",
        "WANDB_CONFIG_DIR": src_root / ".wandb_config",
        "WANDB_DATA_DIR": src_root / ".wandb_data",
    }
    for env_key, path in paths.items():
        os.environ.setdefault(env_key, str(path))
        Path(os.environ[env_key]).mkdir(parents=True, exist_ok=True)


def log_sample_images(dataset="fashion_mnist", samples_per_class=5):
    _configure_local_wandb_dirs()

    run = wandb.init(
        project="da6401-assignment-1",
        name=f"data_exploration_{dataset}",
        tags=["data_exploration"],
        reinit=True,
    )

    if dataset == "mnist":
        (X, y), _ = load_dataset("mnist")
        class_names = MNIST_CLASS_NAMES
    else:
        (X, y), _ = load_dataset("fashion_mnist")
        class_names = FASHION_CLASS_NAMES
    X = X.reshape(-1, 28, 28)

    # Create W&B Table
    columns = ["class_id", "class_name", "image"]
    table = wandb.Table(columns=columns)

    for cls in range(10):
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, samples_per_class, replace=False)
        for idx in selected:
            img = wandb.Image(X[idx])
            table.add_data(int(cls), class_names[cls], img)

    wandb.log({"sample_images": table})
    print(f"Logged {samples_per_class * 10} sample images for {dataset}")
    run.finish()


if __name__ == "__main__":
    log_sample_images("mnist")
    log_sample_images("fashion_mnist")
    print("Done! Check your W&B project for the tables.")
    print("Navigate to: your run page -> click on run name -> look for 'sample_images' table")
