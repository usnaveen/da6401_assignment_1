import argparse
import os

try:
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_dataset
except ModuleNotFoundError:
    from .ann.neural_network import NeuralNetwork
    from .utils.data_loader import load_dataset

try:
    import wandb
except Exception:
    class _WandbStub:
        def init(self, *a, **kw): return None
        def log(self, *a, **kw): return None
        def finish(self, *a, **kw): return None
    wandb = _WandbStub()


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
    p = argparse.ArgumentParser(description="DA6401 Assignment 1 - Training")
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
    return p


def parse_arguments(cli_args=None):
    parser = _build_parser()
    args, unknown = parser.parse_known_args(cli_args)
    if unknown:
        print(f"Warning: ignoring unknown CLI args: {unknown}")
    args.hidden_size = _normalize_hidden_size(args.hidden_size)
    if args.loss == "mean_squared_error":
        args.loss = "mse"
    if args.num_layers is None:
        args.num_layers = len(args.hidden_size.split(","))
    return args


def parse_args():
    return parse_arguments()


def _init_wandb(args):
    mode = os.getenv("WANDB_MODE")
    if mode is None and not os.getenv("WANDB_API_KEY"):
        mode = "disabled"
    try:
        wandb.init(project=args.wandb_project, config=vars(args), mode=mode)
    except Exception:
        os.environ["WANDB_MODE"] = "disabled"
        try:
            wandb.init(project=args.wandb_project, config=vars(args), mode="disabled")
        except Exception:
            pass


if __name__ == "__main__":
    args = parse_arguments()
    (X, y), (X_test, y_test) = load_dataset(args.dataset)

    actual_hidden_count = len(args.hidden_size.split(","))
    if args.num_layers != actual_hidden_count:
        print(f"Warning: --num_layers ({args.num_layers}) doesn't match hidden_size ({actual_hidden_count}). Using {actual_hidden_count}.")
        args.num_layers = actual_hidden_count

    _init_wandb(args)

    model = NeuralNetwork(args)
    model.train(X, y, args.epochs, args.batch_size, X_test, y_test)

    print("\n--- Final Test Evaluation ---")
    model.evaluate(X_test, y_test)

    try:
        wandb.finish()
    except Exception:
        pass
