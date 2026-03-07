"""
Section 2.10 — Fashion-MNIST Transfer Challenge (5 marks)
Run your 3 best MNIST configurations on Fashion-MNIST.

INSTRUCTIONS:
  1. First complete all MNIST experiments and identify your 3 best configs.
  2. Update the CONFIGS list below with your 3 best (architecture + optimizer + activation).
  3. Run: cd src && python -m experiments.fashion_mnist_transfer

Run from project root:
    cd src && python -m experiments.fashion_mnist_transfer
"""
import numpy as np
import wandb
from argparse import Namespace
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


# =============================================================================
# Top 3 configurations from the MNIST hyperparameter sweep
# =============================================================================
CONFIGS = [
    {
        "name": "config1_best_sweep (SGD/Tanh/3L)",
        "optimizer": "sgd", "learning_rate": 0.01,
        "num_layers": 3, "hidden_size": "128,128,128",
        "activation": "tanh", "weight_init": "xavier",
        "batch_size": 32, "weight_decay": 0.0,
    },
    {
        "name": "config2_rmsprop_tanh_3L",
        "optimizer": "rmsprop", "learning_rate": 0.001,
        "num_layers": 3, "hidden_size": "128,128,128",
        "activation": "tanh", "weight_init": "xavier",
        "batch_size": 32, "weight_decay": 0.0,
    },
    {
        "name": "config3_sgd_relu_3L",
        "optimizer": "sgd", "learning_rate": 0.01,
        "num_layers": 3, "hidden_size": "128,128,128",
        "activation": "relu", "weight_init": "xavier",
        "batch_size": 32, "weight_decay": 0.0,
    },
]


def run_experiment(config):
    (X, y), (X_test, y_test) = load_dataset("fashion_mnist")

    run_name = config.pop("name")

    args = Namespace(
        dataset="fashion_mnist",
        epochs=10,
        loss="cross_entropy",
        wandb_project="da6401-assignment-1",
        **config,
    )

    wandb.init(
        project="da6401-assignment-1",
        name=f"fashion_{run_name}",
        tags=["fashion_mnist_transfer"],
        config=vars(args),
    )

    model = NeuralNetwork(args)
    model.train(X, y, args.epochs, args.batch_size, X_test, y_test, save_artifacts=False)

    print(f"\n--- Final Test Evaluation for {run_name} ---")
    acc, prec, rec, f1 = model.evaluate(X_test, y_test)
    wandb.log({"final_test_acc": acc, "final_test_f1": f1})

    wandb.finish()
    return run_name, acc, f1


if __name__ == "__main__":
    results = []

    for config in CONFIGS:
        config_copy = config.copy()
        name = config_copy["name"]
        print(f"\n{'='*60}")
        print(f"Running config: {name} on Fashion-MNIST")
        print(f"{'='*60}")
        name, acc, f1 = run_experiment(config_copy)
        results.append((name, acc, f1))

    print(f"\n{'='*60}")
    print("FASHION-MNIST TRANSFER RESULTS")
    print(f"{'='*60}")
    for name, acc, f1 in results:
        print(f"  {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

    print("\nDone! Compare these with your MNIST results in the W&B report.")
