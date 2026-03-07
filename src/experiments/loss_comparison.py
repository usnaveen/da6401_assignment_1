"""
Section 2.6 — Loss Function Comparison (4 marks)
Compare MSE vs Cross-Entropy with same architecture and LR.

Run from project root:
    cd src && python -m experiments.loss_comparison
"""
import numpy as np
import wandb
from argparse import Namespace
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def run_experiment(loss_fn, run_name):
    (X, y), (X_test, y_test) = load_dataset("mnist")

    args = Namespace(
        dataset="mnist", epochs=10, batch_size=32,
        loss=loss_fn, optimizer="rmsprop",
        learning_rate=0.001, weight_decay=0.0,
        num_layers=2, hidden_size="128,128",
        activation="relu", weight_init="xavier",
        wandb_project="da6401-assignment-1",
    )

    wandb.init(
        project="da6401-assignment-1",
        name=run_name,
        tags=["loss_comparison"],
        config=vars(args),
    )

    model = NeuralNetwork(args)
    model.train(X, y, args.epochs, args.batch_size, X_test, y_test)

    wandb.finish()
    print(f"Completed: {run_name}")


if __name__ == "__main__":
    experiments = [
        ("cross_entropy", "cross_entropy_loss"),
        ("mse",           "mse_loss"),
    ]

    for loss_fn, name in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        run_experiment(loss_fn, name)

    print("\nLoss comparison complete!")
    print("In W&B: filter by tag 'loss_comparison', overlay train_loss and val_acc curves.")
