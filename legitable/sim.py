import numpy as np
# np.seterr(all='raise')
# np.seterr(divide='ignore', invalid='ignore')
# np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
np.set_printoptions(precision=3)

from env.env import Env
from utils.eval import Eval
from utils.config import load_config
from utils.animation import Animate

DEFAULT_CONFIG = "./sim.toml"

def main():
    config = load_config(DEFAULT_CONFIG)
    trial_cnt = config.random_scenarios if config.scenario == "random" else 1
    eval = Eval(config, trial_cnt)
    for i in range(trial_cnt):
        ani = Animate(config)
        for policy_id, ego_policy in enumerate(config.policies):
            np.random.seed(i)
            env = Env(config, ego_policy, policy_id)
            while not env.done:
                env.update()
            env.trim_logs()
            # eval.evaluate(env, i)
            ani.animate(env)
        ani.overlay()
    # eval.get_summary()

if __name__ == '__main__':
    main()
