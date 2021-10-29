import os
import sys
from collections import namedtuple

import tomli

DEFAULT_CONFIG = "../default_config.toml"


def flatten_dict(config):
    r = dict()
    for k, v in config.items():
        if isinstance(v, dict):
            r.update(flatten_dict(v))
        else:
            r[k] = v
    return r


def load_config(config_file):
    config_file = os.path.isfile(config_file) and config_file or DEFAULT_CONFIG
    abs_config = os.path.abspath(config_file)
    try:
        with open(config_file, "rb") as f:
            print(f"Reading config file {abs_config}")
            config = flatten_dict(tomli.load(f))
            return namedtuple("Config", config.keys())(**config)
    except OSError as e:
        print(f"OSError opening {abs_config}")
        print(e)
        sys.exit(os.EX_OSFILE)
