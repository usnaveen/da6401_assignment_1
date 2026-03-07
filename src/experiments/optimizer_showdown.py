"""
Section 2.3 — Optimizer Showdown (5 marks)
Compare SGD, Momentum, NAG, RMSProp with same architecture.
Architecture: 3 hidden layers, 128 neurons each, ReLU activation.

Run from project root:
    cd src && python -m experiments.optimizer_showdown
"""
import subprocess
import sys

COMMON = [
    "-d", "mnist",
    "-e", "10",
    "-b", "32",
    "-nhl", "3",
    "-sz", "128,128,128",
    "-a", "relu",
    "-w_i", "xavier",
    "-l", "cross_entropy",
    "-wd", "0.0",
    "-w_p", "da6401-assignment-1",
]

EXPERIMENTS = [
    {"optimizer": "sgd",      "lr": "0.01"},
    {"optimizer": "momentum", "lr": "0.01"},
    {"optimizer": "nag",      "lr": "0.01"},
    {"optimizer": "rmsprop",  "lr": "0.001"},
]

if __name__ == "__main__":
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Running optimizer: {exp['optimizer']}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "train.py",
            "-o", exp["optimizer"],
            "-lr", exp["lr"],
        ] + COMMON

        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            print(f"ERROR: {exp['optimizer']} run failed!")
        else:
            print(f"Completed: {exp['optimizer']}")

    print("\nAll optimizer runs complete! Check W&B for comparison plots.")
