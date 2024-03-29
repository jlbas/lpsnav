import numpy as np
from utils.utils import val_as_list
from utils import helper
from utils import utils


class AttemptsExceededError(Exception):
    def __init__(self, max_attempts, message=None):
        self.max_attempts = max_attempts
        self.default_msg = (
            f"Couldn't create feasible configuration within {self.max_attempts} attempts"
        )
        self.message = self.default_msg if message is None else message
        super().__init__(self.message)


def is_feasible(positions, min_dist):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if helper.dist(positions[i], positions[j]) < min_dist:
                return False
    return True


def init_scenario(base_id, s_conf, e_conf, a_conf, rng):
    starts, goals, max_speeds = get_init_configuration(s_conf, e_conf, a_conf, rng)
    other_policy = val_as_list(s_conf["policy"] if s_conf["homogeneous"] else s_conf["human_model"])
    policies = [s_conf["policy"]] + list(rng.choice(other_policy, len(starts) - 1))
    ids = range(base_id * len(policies), (base_id + 1) * len(policies))
    agents = {}
    for i, (id, s, g, v_max, p) in enumerate(zip(ids, starts, goals, max_speeds, policies)):
        is_ego = i == 0
        merged_conf = {**{k: v for k, v in a_conf.items() if not isinstance(v, dict)}, **a_conf[p]}
        agents[id] = utils.get_cls("policies", p)(merged_conf, id, p, is_ego, v_max, s, g, rng)
    return agents


def get_random_pos(rng, w, h, n):
    return np.column_stack((w * (rng.random(n) - 0.5), h * (rng.random(n) - 0.5)))


def get_init_configuration(s_conf, e_conf, a_conf, rng):
    min_dist = 2 * a_conf["radius"] + e_conf["min_start_buffer"]
    if s_conf["name"] == "predefined":
        for _ in range(e_conf["max_init_attempts"]):
            x = s_conf["long_dist"] / 2
            y = 2 * a_conf["radius"] + s_conf["lat_dist"]
            max_speeds = np.array([])
            if s_conf["configuration"] == "swap":
                starts = np.array([[-x, 0], [x, 0]])
                goals = starts.copy()[::-1]
            elif s_conf["configuration"] == "pass":
                starts = np.array([[-x, -y / 2], [x, y / 2]])
                goals = np.array([-1, 1]) * starts
            elif s_conf["configuration"] == "acute":
                starts = np.array([[-x, 0], [-x + 0.5, -y]])
                goals = -starts.copy()
            elif s_conf["configuration"] == "obtuse":
                starts = np.array([[-x, 0], [x - 0.5, y]])
                goals = -starts.copy()
            elif s_conf["configuration"] == "split":
                starts = np.array([[-x, 0], [x, y], [x, -y]])
                goals = np.array([-1, 1]) * starts
            elif s_conf["configuration"] == "t_junction":
                starts = np.array([[-x, 0], [0, -x]])
                goals = np.array([[-1, 1], [1, -1]]) * starts
            elif s_conf["configuration"] == "2_agent_t_junction":
                starts = np.array([[-x, 0], [-y / 2, x], [y / 2, -x]])
                goals = np.array([[-1, 1], [1, -1], [1, -1]]) * starts
            elif s_conf["configuration"] == "overtake":
                starts = np.array([[-x - 1, 0], [-x + 1, 0]])
                goals = np.array([-1, 1]) * starts
                max_speeds = np.array([a_conf["max_speed"], 0.4 * a_conf["max_speed"]])
            else:
                raise ValueError(f"Scenario {s_conf['configuration']} is not recognized")
            starts += rng.uniform(-s_conf["uniform_bnd"], s_conf["uniform_bnd"], np.shape(starts))
            goals += rng.uniform(-s_conf["uniform_bnd"], s_conf["uniform_bnd"], np.shape(goals))
            if max_speeds.size == 0:
                max_speeds = np.full(len(starts), a_conf["max_speed"])
            if is_feasible(starts, min_dist) and is_feasible(goals, min_dist):
                break
        else:
            raise AttemptsExceededError(e_conf["max_init_attempts"])
    elif s_conf["name"] == "random":
        max_speeds = rng.normal(
            s_conf["des_speed_mean"], s_conf["des_speed_std_dev"], size=s_conf["number_of_agents"]
        )
        for _ in range(e_conf["max_init_attempts"]):
            starts = get_random_pos(rng, s_conf["width"], s_conf["height"], s_conf["number_of_agents"])
            goals = get_random_pos(rng, s_conf["width"], s_conf["height"], s_conf["number_of_agents"])
            feasible = is_feasible(starts, min_dist) and is_feasible(goals, min_dist)
            far_enough = np.all(helper.dist(starts, goals) > s_conf["min_start_goal_sep"])
            if feasible and far_enough:
                break
        else:
            raise AttemptsExceededError(e_conf["max_init_attempts"])
    elif s_conf["name"] == "circle":
        thetas = np.linspace(
            0, 2 * np.pi * (1 - 1 / s_conf["number_of_agents"]), s_conf["number_of_agents"]
        )
        starts = s_conf["radius"] * helper.vec(thetas)
        goals = -starts
        max_speeds = np.full(len(starts), a_conf["max_speed"])
    elif s_conf["name"] == "custom":
        starts = np.array(s_conf["starts"])
        goals = np.array(s_conf["goals"])
        max_speeds = np.array(s_conf.get("max_speeds", len(starts) * [a_conf["max_speed"]]))
    else:
        raise ValueError(f"Scenario {s_conf['name']} is not recognized")
    return starts, goals, max_speeds
