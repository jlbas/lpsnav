import ast
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


def init_agents(config, env, rng, ego_policy, scenario, num_of_agents, iter):
    starts, goals, max_speeds = get_init_configs(config, rng, scenario, num_of_agents)
    agents = dict()
    other_policy = ego_policy if config.env.homogeneous else config.env.human_policy
    policies = [ego_policy] + (len(starts) - 1) * [other_policy]
    ids = range(iter * len(policies), iter * len(policies) + len(policies))
    for i, (id, s, g, v_max, p) in enumerate(zip(ids, starts, goals, max_speeds, policies)):
        module = importlib.import_module(f"policies.{p}")
        cls = getattr(module, "".join(wd.capitalize() for wd in p.split("_")))
        is_ego = (i == 0)
        agents[id] = cls(config, env, id, p, is_ego, s, goal=g, max_speed=v_max)
    return agents[ids[0]], agents


def get_init_configs(config, rng, scenario, num_of_agents):
    if scenario == "custom":
        starts, goals = np.swapaxes(config.env.custom_pos, 0, 1)
        max_speeds = (
            config.env.custom_speed
            if isinstance(config.env.custom_speed, list)
            else np.full(len(config.env.custom_pos), config.env.custom_speed)
        )
    elif scenario == "random":
        max_speeds = rng.uniform(*config.env.speed_range, size=num_of_agents)
        min_dist = 2 * config.agent.radius + config.env.min_start_buffer
        for _ in range(100):
            starts = config.env.workspace_length * rng.random((num_of_agents, 2))
            goals = config.env.workspace_length * rng.random((num_of_agents, 2))
            feasible = is_feasible(starts, min_dist) and is_feasible(goals, min_dist)
            far_enough = np.all(helper.dist(starts, goals) > min_dist)
            if feasible and far_enough:
                break
        else:
            raise AttemptsExceededError(100)
    else:
        start_sep = config.env.interaction_dist / 2
        lat_sep = 2 * config.agent.radius + config.env.lat_dist
        if scenario == "swap":
            starts = np.array([[-start_sep, 0], [start_sep, 0]])
            goals = starts[::-1]
        elif scenario == "passing":
            starts = np.array([[-start_sep, 0], [start_sep, lat_sep]])
            goals = np.array([-1, 1]) * starts
        elif scenario == "acute":
            starts = np.array([[-start_sep, 0], [-start_sep + 0.5, -lat_sep]])
            goals = -starts
        elif scenario == "obtuse":
            starts = np.array([[-start_sep, 0], [start_sep - 0.5, lat_sep]])
            goals = -starts
        elif scenario == "2_agent_split":
            starts = np.array([[-start_sep, 0], [start_sep, lat_sep], [start_sep, -lat_sep]])
            goals = np.array([-1, 1]) * starts
        elif scenario == "t_junction":
            starts = [[-start_sep, 0], [0, -start_sep]]
            goals = np.array([[-1, 1], [1, -1]]) * starts
        elif scenario == "2_agent_t_junction":
            starts = np.array([[-start_sep, 0], [-lat_sep / 2, start_sep], [lat_sep / 2, -start_sep]])
            goals = np.array([[-1, 1], [1, -1], [1, -1]]) * starts
        elif scenario == "frp":
            y_max = 0.5 * lat_sep * (config.env.num_of_agents - 1)
            y_starts = np.linspace(-y_max, y_max, config.env.num_of_agents)
            starts = np.column_stack((np.full(y_starts.shape, start_sep), y_starts))
            starts = np.vstack(([-start_sep, 0], starts))
            goals = np.array([-1, 1]) * starts
        elif scenario == "circle":
            agent_cnt = num_of_agents
            thetas = np.linspace(0, 2 * np.pi * (1 - 1 / agent_cnt), agent_cnt)
            starts = config.env.circle_radius * helper.vec(thetas)
            goals = -starts
        else:
            raise ValueError(f"Scenario '{scenario}' is not recognized")
        max_speeds = np.full(len(starts), config.agent.max_speed)

    # Without noise, RVO might not work properly
    bound = 0.01
    starts += rng.uniform(-bound, bound, np.shape(starts))
    goals += rng.uniform(-bound, bound, np.shape(goals))

    return starts, goals, max_speeds
