import tomli
from collections import namedtuple

def flatten_dict(config):
    r = dict()
    for k, v in config.items():
        if isinstance(v, dict):
            r.update(flatten_dict(v))
        else:
            r[k] = v
    return r

def load_config(default_config):
    with open(default_config, "rb") as f:
        config = flatten_dict(tomli.load(f))
    return namedtuple('Config', config.keys())(**config)
