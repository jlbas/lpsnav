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


def get_random_pos(rng, length, width=None, num=1):
    if width is None:
        return length * (rng.random((num, 2)) - 0.5)
    return np.column_stack((length * (rng.random(num) - 0.5), width * (rng.random(num) - 0.5)))


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
        starts += rng.normal(scale=s_conf["scale"], size=np.shape(starts))
        goals += rng.normal(scale=s_conf["scale"], size=np.shape(goals))
        max_speeds = np.full(len(starts), a_conf["max_speed"])
    elif s_conf["name"] == "random":
        max_speeds = rng.normal(
            s_conf["des_speed_mean"], s_conf["des_speed_std_dev"], size=s_conf["number_of_agents"]
        )
        min_dist = 2 * a_conf["radius"] + s_conf["min_start_buffer"]
        for _ in range(s_conf["max_init_attempts"]):
            starts = get_random_pos(rng, s_conf["length"], num=s_conf["number_of_agents"])
            goals = get_random_pos(rng, s_conf["length"], num=s_conf["number_of_agents"])
            feasible = is_feasible(starts, min_dist) and is_feasible(goals, min_dist)
            far_enough = np.all(helper.dist(starts, goals) > s_conf["min_start_goal_sep"])
            if feasible and far_enough:
                break
        else:
            raise AttemptsExceededError(s_conf["max_init_attempts"])
    elif s_conf["name"] == "hallway":
        x = s_conf["length"]
        y = s_conf["width"] / 2
        n = 1 + s_conf["par_agents"] + s_conf["perp_agents"]
        max_speeds = rng.normal(s_conf["des_speed_mean"], s_conf["des_speed_std_dev"], size=n)
        min_dist = 2 * a_conf["radius"] + s_conf["min_start_buffer"]
        lb = s_conf["length"] - 2 * a_conf["radius"]
        wb = s_conf["width"] - 2 * a_conf["radius"]
        for _ in range(s_conf["max_init_attempts"]):
            ego_start = np.array([[-1.5 * s_conf["length"], 0]])
            par_starts = get_random_pos(rng, lb, wb, s_conf["par_agents"])
            perp_starts = np.sort(get_random_pos(rng, lb, 1, s_conf["perp_agents"]), axis=0)
            perp_starts[:,1] += helper.dist(ego_start, perp_starts)
            perp_starts[:,1] *= rng.choice([1, -1], s_conf["perp_agents"])
            starts = np.concatenate((ego_start, par_starts, perp_starts))
            par_goals = par_starts - np.array([s_conf["length"], 0])
            perp_goals = perp_starts * np.array([1, -1])
            ego_goal = np.array([[s_conf["length"] / 2, 0]])
            goals = np.concatenate((ego_goal, par_goals, perp_goals))
            if is_feasible(starts, min_dist) and is_feasible(goals, min_dist):
                if s_conf["walls"]:
                    if s_conf["perp_agents"]:
                        x_offset = perp_starts[0][0] - s_conf["door_width"] / 2
                        wall = np.array([[-3 * x / 2, y], [x_offset, y]])
                        s_walls = np.tile(wall, (2 * (s_conf["perp_agents"] + 1), 1, 1))
                        for i in range(1, s_conf["perp_agents"]):
                            pt0 = [s_walls[i-1][1][0] + s_conf["door_width"], y]
                            pt1 = [perp_starts[i][0] - s_conf["door_width"] / 2, y]
                            s_walls[i] = np.array([pt0, pt1])
                        pt0 = [s_walls[s_conf["perp_agents"]-1][1][0] + s_conf["door_width"], y]
                        pt1 = [x / 2, y]
                        s_walls[s_conf["perp_agents"]] = np.array([pt0, pt1])
                        mirrored_walls = np.array([1, -1]) * s_walls[:s_conf["perp_agents"]+1]
                        s_walls[s_conf["perp_agents"]+1:] = mirrored_walls
                    else:
                        s_walls = np.array([[[-x / 2, y], [x / 2, y]], [[-x / 2, -y], [x / 2, -y]]])
                    walls = np.vstack((walls, s_walls)) if np.any(walls) else s_walls.copy()
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
