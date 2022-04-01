import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Run time arguments")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.toml",
        help="Location of the configuration file",
    )
    return parser
