"""
Section 2.9 — Weight Initialization & Symmetry (7 marks)
Compare Zero vs Xavier init. Log per-neuron gradients (5 neurons, same layer)
over the first 50 training iterations.

Run from project root:
    cd src && python -m experiments.weight_init_symmetry
"""
import numpy as np
import wandb
from argparse import Namespace
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def run_experiment(weight_init, run_name):
    (X, y), (X_test, y_test) = load_dataset("mnist")

    args = Namespace(
        dataset="mnist", epochs=5, batch_size=32,
        loss="cross_entropy", optimizer="sgd",
        learning_rate=0.01, weight_decay=0.0,
        num_layers=2, hidden_size="128,128",
        activation="relu", weight_init=weight_init,
        wandb_project="da6401-assignment-1",
    )

    wandb.init(
        project="da6401-assignment-1",
        name=run_name,
        tags=["weight_init_symmetry"],
        config=vars(args),
    )

    model = NeuralNetwork(args)

    # Track 5 neurons in the FIRST hidden layer (layer 0)
    model.train(
        X, y, args.epochs, args.batch_size,
        X_test, y_test,
        log_neuron_gradients=True,  # <-- logs per-neuron gradients
        neuron_layer_idx=0,         # first hidden layer
        neuron_indices=[0, 1, 2, 3, 4],  # 5 neurons
        max_iterations=50,          # stop after 50 iterations
    )

    wandb.finish()
    print(f"Completed: {run_name}")


if __name__ == "__main__":
    experiments = [
        ("zero",   "zero_init_symmetry"),
        ("xavier", "xavier_init_symmetry"),
    ]

    for w_init, name in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name} (weight_init={w_init})")
        print(f"{'='*60}")
        run_experiment(w_init, name)

    print("\nWeight init symmetry experiments complete!")
    print("In W&B: filter by tag 'weight_init_symmetry'.")
    print("Plot 'layer0_neuron0_grad_norm' through 'layer0_neuron4_grad_norm'.")
    print("For zero init: all 5 lines should overlap perfectly (symmetry).")
    print("For xavier init: 5 distinct gradient trajectories.")
