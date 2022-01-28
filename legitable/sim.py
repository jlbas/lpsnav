import numpy as np
from env.env import Env
from utils.animation import Animate
from utils.config import load_config
from utils.eval import Eval
from utils.utils import get_args

np.seterr(all="raise")
# np.seterr(divide='ignore', invalid='ignore')
# np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
np.set_printoptions(precision=5)


def main():
    args = get_args()
    config = load_config(args.config)
    num_of_agents_lst = [
        config.env.num_of_agents
        if s in ("circle", "random")
        else [len(config.env.custom_pos)]
        if s == "custom"
        else [-1]
        for s in config.env.scenarios
    ]
    trial_cnts = [config.env.random_scenarios if s == "random" else 1 for s in config.env.scenarios]
    eval = Eval(config, num_of_agents_lst, trial_cnts)
    for scenario, num_of_agents, trial_cnt in zip(
        config.env.scenarios, num_of_agents_lst, trial_cnts
    ):
        for n in num_of_agents:
            for i in range(trial_cnt):
                print(f"ITERATION {i}")
                ani = Animate(config, scenario, i)
                for policy_id, ego_policy in enumerate(config.env.policies):
                    rng = np.random.default_rng(i)
                    env = Env(config, rng, ego_policy, i, scenario, n, policy_id)
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
