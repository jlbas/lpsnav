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
    for scenario in config.scenarios:
        trial_cnt = config.random_scenarios if scenario == "random" else 1
        eval = Eval(trial_cnt, config, scenario)
        for i in range(trial_cnt):
            print(f"ITERATION {i}")
            ani = Animate(config, scenario)
            for policy_id, ego_policy in enumerate(config.policies):
                np.random.seed(i)
                env = Env(config, ego_policy, i, scenario, policy_id)
                while not env.done:
                    env.update()
                env.trim_logs()
                eval.evaluate(i, env)
                ani.animate(i, env, eval)
            if config.overlay and len(config.policies) > 1:
                ani.overlay()
        eval.get_summary()


if __name__ == "__main__":
    main()
