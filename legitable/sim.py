import numpy as np
from env.env import Env
from utils.animation import Animate
from utils.config import load_config
from utils.eval import Eval
from utils.utils import get_args
import logging
import logging.config

np.seterr(all="raise")
# np.seterr(divide='ignore', invalid='ignore')
# np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
np.set_printoptions(precision=5)


def main():
    args = get_args()
    config = load_config(args.config)
    logging.config.fileConfig("logging.conf", defaults={"logFile": config.log_file})
    logger = logging.getLogger(__name__)
    n_ws_lst = [
        [config.env.num_of_agents, config.env.workspace_length]
        if s in ("circle", "random")
        else [[len(config.env.custom_pos)], [config.env.circle_radius]]
        if s == "custom"
        else [[-1], [-1]]
        for s in config.env.scenarios
    ]
    iter_lst = [config.env.random_scenarios if s == "random" else 1 for s in config.env.scenarios]
    eval = Eval(config, [i[0] for i in n_ws_lst], iter_lst)
    for scenario, n_ws, iters in zip(config.env.scenarios, n_ws_lst, iter_lst):
        for n, ws in zip(*n_ws):
            for i in range(iters):
                logger.info(f"{scenario} {n} agent iteration {i}")
                ani = Animate(config, scenario, i)
                for policy_id, ego_policy in enumerate(config.env.policies):
                    rng = np.random.default_rng(i)
                    env = Env(config, rng, ego_policy, i, scenario, n, ws, config.env.sg_ws_ratio, policy_id)
                    while not env.done:
                        env.update()
                    env.trim_logs()
                    eval.evaluate(i, env.dt, env.ego_agent, scenario, n)
                    ani.animate(i, env.agents, env.ego_agent, scenario, n, str(env), eval)
                if config.animation.overlay and len(config.env.policies) > 1:
                    ani.overlay()
    eval.get_summary()


if __name__ == "__main__":
    main()
