import os
import tomli


def load_config(fname):
    with open(fname, "rb") as f:
        config = tomli.load(f)
        print(f"Reading config file {os.path.abspath(fname)}")
        return config
