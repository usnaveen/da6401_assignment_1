from runpy import run_module

from src.inference import parse_args, parse_arguments  # noqa: F401


if __name__ == "__main__":
    run_module("src.inference", run_name="__main__")

