import argparse
import numpy as np
import json
from argparse import Namespace

try:
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import get_artifact_path, load_dataset
except ModuleNotFoundError:
    from .ann.neural_network import NeuralNetwork
    from .utils.data_loader import get_artifact_path, load_dataset


def _normalize_hidden_size(val):
    if isinstance(val, str):
        return val.replace(" ", "")
    if isinstance(val, (list, tuple)):
        parts = []
        for item in val:
            s = str(item).strip()
            if "," in s:
                parts.extend([p.strip() for p in s.split(",") if p.strip()])
            elif s:
                parts.append(s)
        return ",".join(parts) if parts else "128,128"
    return str(val)


def _build_parser():
    p = argparse.ArgumentParser(description="DA6401 Assignment 1 - Inference")
    p.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e", "--epochs", type=int, default=10)
    p.add_argument("-b", "--batch_size", "--batch-size", type=int, default=32)
    p.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "mse"])
    p.add_argument("-o", "--optimizer", type=str, default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd", "--weight_decay", type=float, default=0.0001)
    p.add_argument("-nhl", "--num_layers", "--num_hidden_layers", type=int, default=None)
    p.add_argument("-sz", "--hidden_size", "--hidden_layer_size", "--hidden_layer_sizes", nargs="+", default=["128,128"])
    p.add_argument("-a", "--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    p.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "xavier"])
    p.add_argument("-w_p", "--wandb_project", type=str, default="da6401-assignment-1")
    p.add_argument("--model_path", type=str, default=str(get_artifact_path("best_model.npy")))
    return p


def parse_arguments(cli_args=None):
    parser = _build_parser()
    args, unknown = parser.parse_known_args(cli_args)
    if unknown:
        print(f"Warning: ignoring unknown CLI args: {unknown}")
    args.hidden_size = _normalize_hidden_size(args.hidden_size)
    if args.num_layers is None:
        args.num_layers = len(args.hidden_size.split(","))
    return args


def parse_args():
    return parse_arguments()


if __name__ == "__main__":
    args = parse_arguments()
    config_path = get_artifact_path("best_config.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        model_path = args.model_path
        args = Namespace(**config)
        args.model_path = model_path
    except FileNotFoundError:
        print("Using CLI defaults - best_config.json not found")

    _, (X_test, y_test) = load_dataset(args.dataset)
    model = NeuralNetwork(args)

    try:
        weights = np.load(args.model_path, allow_pickle=True).item()
        model.set_weights(weights)
    except FileNotFoundError:
        print(f"{args.model_path} not found — run train.py first")
        exit(1)

    print("--- Test Set Evaluation ---")
    model.evaluate(X_test, y_test)
