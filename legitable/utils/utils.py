import argparse


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
