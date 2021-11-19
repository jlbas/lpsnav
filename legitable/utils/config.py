import os
import sys
from collections import namedtuple

import tomli

DEFAULT_CONFIG = "../default_config.toml"


def namedtuple_from_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = namedtuple_from_dict(v)
    return namedtuple("obj", d.keys())(*d.values())


def load_config(config_file):
    config_file = os.path.isfile(config_file) and config_file or DEFAULT_CONFIG
    abs_config = os.path.abspath(config_file)
    try:
        with open(config_file, "rb") as f:
            print(f"Reading config file {abs_config}")
            config = tomli.load(f)
            for policy in config["env"]["policies"]:
                if policy not in config:
                    config[policy] = dict()
                for k, v in config["agent"].items():
                    if k not in config[policy]:
                        config[policy][k] = v
            return namedtuple_from_dict(config)
    except OSError as e:
        print(f"OSError opening {abs_config}")
        print(e)
        sys.exit(os.EX_OSFILE)
