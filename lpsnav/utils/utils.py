from collections.abc import Iterable


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
    s_conf = config["scenario"][scenario].copy()
    opts, n_opts = repeat_to_match(*s_conf.values())
    s_conf.update(dict(zip(s_conf.keys(), opts)))
    for i in range(n_opts):
        iters = s_conf.get("iters", [1])[0]
        for j in range(iters):
            for p in val_as_list(config["scenario"]["policy"]):
                d = {
                    "i": i * iters + j,
                    "iter": j,
                    "policy": p,
                    **{k: v[i] for k, v in s_conf.items()},
                }
                for k, v in top_level_conf.items():
                    d.setdefault(k, v)
                s_configs.append(d)
    return s_configs


def flatten(list_of_lists):
    for list in list_of_lists:
        if isinstance(list, Iterable) and not isinstance(list, (str, bytes)):
            yield from flatten(list)
        else:
            yield list
