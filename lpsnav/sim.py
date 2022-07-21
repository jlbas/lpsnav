import numpy as np
import os
from env.env import Env
from env.scenario_factory import init_scenario
from utils.animation import Animate
from utils.config import load_config
from utils.eval import Eval
from utils.parser import get_parser
from utils.utils import format_scenarios, val_as_list, get_scenario_name
import logging
import logging.config
from tqdm import tqdm


def run(s_name, config, s_configs):
    eval = Eval(config, s_configs)
    ani = Animate(config["animation"])
    p_cnt = len(val_as_list(config["scenario"]["policy"]))
    overlay = p_cnt > 1 and config["animation"]["overlay"]
    for i, s_conf in enumerate(tqdm(s_configs) if config["progress_bar"] else s_configs):
        rng = np.random.default_rng(s_conf["iter"])
        agents, walls = init_scenario(i, s_conf, config["env"], config["agent"], rng)
        env = Env(config["env"], agents, walls)
        while env.is_running():
            env.update()
        env.trim_logs()
        fname = get_scenario_name(s_name, s_conf)
        eval.evaluate(i, env, fname)
        ani.animate(env.dt, env.agents, env.logs, env.walls, fname, overlay, env.ego_id)
        if overlay and i % p_cnt == p_cnt - 1:
            ani.overlay(env.dt, walls, fname.replace(s_conf["policy"], "overlay"))
    fname = get_scenario_name(s_name, config)
    eval.get_summary(fname)


def main():
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    os.makedirs(config["log_dir"], exist_ok=True)
    buf = os.path.join(config["log_dir"], "log")
    logging.config.fileConfig("config/logging.conf", defaults={"logFile": buf})

    for s_name in val_as_list(config["scenario"]["name"]):
        s_configs = format_scenarios(s_name, config)
        run(s_name, config, s_configs)
    print("Completed all scenarios")


if __name__ == "__main__":
    main()
