from runpy import run_module

from src.train import parse_args, parse_arguments  # noqa: F401


if __name__ == "__main__":
    run_module("src.train", run_name="__main__")

