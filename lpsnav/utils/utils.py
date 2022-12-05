from collections.abc import Iterable
import importlib
import itertools


def val_as_list(val):
    return val if isinstance(val, list) else [val]


def repeat_to_match(*vals):
    vals = [val_as_list(v) for v in vals]
    val_lens = [len(v) for v in vals]
    n_opts = min([v for v in val_lens if v > 1], default=1)
    return [n_opts * v if v_len == 1 else v[:n_opts] for v, v_len in zip(vals, val_lens)], n_opts


def format_scenarios(scenario, config):
    s_configs = []
    top_level_conf = {k: v for k, v in config["scenario"].items() if not isinstance(v, dict)}
    s_conf = {k: val_as_list(v) for k, v in config["scenario"][scenario].items()}
    if scenario == "custom":
        conf_combos = [list(s_conf.values())]
    else:
        conf_combos = list(itertools.product(*s_conf.values()))
    iters = s_conf.get("iters", [1])[0]
    for i, conf in enumerate(conf_combos):
        for j in range(iters):
            for p in val_as_list(config["scenario"]["policy"]):
                d = {
                    "i": i * iters + j,
                    "iter": j,
                    **top_level_conf,
                    "policy": p,
                    **dict(zip(s_conf.keys(), conf)),
                }
                s_configs.append(d)
    return s_configs


def flatten(list_of_lists):
    for list in list_of_lists:
        if isinstance(list, Iterable) and not isinstance(list, (str, bytes)):
            yield from flatten(list)
        else:
            yield list


def get_cls(abs_path, class_name):
    try:
        module = importlib.import_module(f"{abs_path}.{class_name}")
    except ModuleNotFoundError:
        module = importlib.import_module(abs_path)
    return getattr(module, "".join(wd.capitalize() for wd in class_name.split("_")))


def get_fname(s_name, s_conf):
    param = s_conf.get('comparison_param'), '0'
    iter = s_conf.get('iter', '0')
    return f"{s_name}_{s_conf.get(param)}_iter_{iter}_{s_conf['policy']}"
