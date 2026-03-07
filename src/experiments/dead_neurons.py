"""
Section 2.5 — Dead Neuron Investigation (6 marks)
ReLU + high LR (0.1) vs Tanh + high LR.
Monitor activation distributions to find dead neurons.

Run from project root:
    cd src && python -m experiments.dead_neurons
"""
import numpy as np
import wandb
from argparse import Namespace
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def run_experiment(activation, learning_rate, run_name):
    (X, y), (X_test, y_test) = load_dataset("mnist")

    args = Namespace(
        dataset="mnist", epochs=10, batch_size=32,
        loss="cross_entropy", optimizer="sgd",
        learning_rate=learning_rate, weight_decay=0.0,
        num_layers=3, hidden_size="128,128,128",
        activation=activation, weight_init="xavier",
        wandb_project="da6401-assignment-1",
    )

    wandb.init(
        project="da6401-assignment-1",
        name=run_name,
        tags=["dead_neurons"],
        config=vars(args),
    )

    model = NeuralNetwork(args)
    model.train(
        X, y, args.epochs, args.batch_size,
        X_test, y_test,
        log_activations=True,  # <-- logs dead neuron fractions per layer
    )

    wandb.finish()
    print(f"Completed: {run_name}")


if __name__ == "__main__":
    experiments = [
        ("relu", 0.1,  "relu_high_lr"),
        ("relu", 0.01, "relu_normal_lr"),
        ("tanh", 0.1,  "tanh_high_lr"),
        ("tanh", 0.01, "tanh_normal_lr"),
    ]

    for act, lr, name in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name} (activation={act}, lr={lr})")
        print(f"{'='*60}")
        run_experiment(act, lr, name)

    print("\nAll dead neuron experiments complete!")
    print("In W&B: filter by tag 'dead_neurons'.")
    print("Plot 'dead_neuron_frac_layer_0/1/2' and 'zero_activation_frac_layer_0/1/2'.")
