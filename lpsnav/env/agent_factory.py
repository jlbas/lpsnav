import importlib

import numpy as np
from utils import helper


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
    starts, goals, max_speeds, walls = get_init_configuration(s_conf, e_conf, a_conf, rng)
    other_policy = s_conf["policy"] if s_conf["homogeneous"] else s_conf["human_model"]
    policies = [s_conf["policy"]] + (len(starts) - 1) * [other_policy]
    ids = range(base_id * len(policies), (base_id + 1) * len(policies))
    agents = {}
    for i, (id, s, g, v_max, p) in enumerate(zip(ids, starts, goals, max_speeds, policies)):
        module = importlib.import_module(f"policies.{p}")
        cls = getattr(module, "".join(wd.capitalize() for wd in p.split("_")))
        is_ego = i == 0
        merged_conf = {**{k: v for k, v in a_conf.items() if not isinstance(v, dict)}, **a_conf[p]}
        agents[id] = cls(merged_conf, id, p, is_ego, v_max, s, g, rng)
    return agents, walls


def get_init_configuration(s_conf, e_conf, a_conf, rng):
    walls = np.array(e_conf.get("walls", []), dtype="float64")
    if s_conf["name"] == "predefined":
        x = s_conf["long_dist"] / 2
        y = 2 * a_conf["radius"] + s_conf["lat_dist"]
        if s_conf["configuration"] == "swap":
            starts = np.array([[-x, 0], [x, 0]])
            goals = starts[::-1]
        elif s_conf["configuration"] == "pass":
            starts = np.array([[-x, 0], [x, y]])
            goals = np.array([-1, 1]) * starts
        elif s_conf["configuration"] == "acute":
            starts = np.array([[-x, 0], [-x + 0.5, -y]])
            goals = -starts
        elif s_conf["configuration"] == "obtuse":
            starts = np.array([[-x, 0], [x - 0.5, y]])
            goals = -starts
        elif s_conf["configuration"] == "split":
            starts = np.array([[-x, 0], [x, y], [x, -y]])
            goals = np.array([-1, 1]) * starts
        elif s_conf["configuration"] == "t_junction":
            starts = [[-x, 0], [0, -x]]
            goals = np.array([[-1, 1], [1, -1]]) * starts
        elif s_conf["configuration"] == "2_agent_t_junction":
            starts = np.array([[-x, 0], [-y / 2, x], [y / 2, -x]])
            goals = np.array([[-1, 1], [1, -1], [1, -1]]) * starts
        else:
            raise ValueError(f"Scenario {s_conf['configuration']} is not recognized")
        max_speeds = np.full(len(starts), a_conf["max_speed"])
    elif s_conf["name"] == "random":
        max_speeds = rng.uniform(
            s_conf["min_des_speed"], s_conf["max_des_speed"], size=s_conf["number_of_agents"]
        )
        min_dist = 2 * a_conf["radius"] + s_conf["min_start_buffer"]
        for _ in range(s_conf["max_init_attempts"]):
            starts = s_conf["workspace_length"] * rng.random((s_conf["number_of_agents"], 2))
            goals = s_conf["workspace_length"] * rng.random((s_conf["number_of_agents"], 2))
            feasible = is_feasible(starts, min_dist) and is_feasible(goals, min_dist)
            far_enough = np.all(helper.dist(starts[0], goals[0]) > s_conf["min_start_goal_sep"])
            if feasible and far_enough:
                break
        else:
            raise AttemptsExceededError(s_conf["max_init_attempts"])
    elif s_conf["name"] == "circle":
        thetas = np.linspace(
            0, 2 * np.pi * (1 - 1 / s_conf["number_of_agents"]), s_conf["number_of_agents"]
        )
        starts = s_conf["radius"] * helper.vec(thetas)
        goals = -starts
        max_speeds = np.full(len(starts), a_conf["max_speed"])
    else:
        raise ValueError(f"Scenario {s_conf['name']} is not recognized")
    bound = 0.01
    starts += rng.uniform(-bound, bound, np.shape(starts))
    goals += rng.uniform(-bound, bound, np.shape(goals))
    return starts, goals, max_speeds, walls
