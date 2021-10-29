import argparse
import sys


def print_to_file(file, message):
    original_stdout = sys.stdout
    with open(file, "w") as outfile:
        sys.stdout = outfile
        print(message)
        sys.stdout = original_stdout


def get_args():
    args = argparse.ArgumentParser(description="Run time arguments")

    args.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config.toml",
        help="Specify the location of the config file",
    )

    return args.parse_args()
