"""
Section 2.4 — Vanishing Gradient Analysis (5 marks)
Fix optimizer to RMSProp. Compare Sigmoid vs ReLU for different depths.
Log gradient norms for first hidden layer.

Run from project root:
    cd src && python -m experiments.vanishing_gradient
"""
import numpy as np
import wandb
from argparse import Namespace
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def run_experiment(activation, num_layers, hidden_size, run_name):
    (X, y), (X_test, y_test) = load_dataset("mnist")

    args = Namespace(
        dataset="mnist", epochs=10, batch_size=32,
        loss="cross_entropy", optimizer="rmsprop",
        learning_rate=0.001, weight_decay=0.0,
        num_layers=num_layers, hidden_size=hidden_size,
        activation=activation, weight_init="xavier",
        wandb_project="da6401-assignment-1",
    )

    wandb.init(
        project="da6401-assignment-1",
        name=run_name,
        tags=["vanishing_gradient"],
        config=vars(args),
    )

    model = NeuralNetwork(args)
    model.train(
        X, y, args.epochs, args.batch_size,
        X_test, y_test,
        log_gradients=True,  # <-- logs gradient norms per layer
    )

    wandb.finish()
    print(f"Completed: {run_name}")


if __name__ == "__main__":
    experiments = [
        ("sigmoid", 2, "128,128",             "sigmoid_2layers"),
        ("relu",    2, "128,128",             "relu_2layers"),
        ("sigmoid", 5, "128,128,128,128,128", "sigmoid_5layers"),
        ("relu",    5, "128,128,128,128,128", "relu_5layers"),
    ]

    for act, nhl, hsz, name in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        run_experiment(act, nhl, hsz, name)

    print("\nAll vanishing gradient experiments complete!")
    print("In W&B: filter by tag 'vanishing_gradient', plot 'grad_norm_layer_0' over iterations.")
