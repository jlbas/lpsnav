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


def far_enough(starts, goals):
    return np.all(helper.dist(starts, goals) > 2)


def init_agents(config, env, rng, ego_policy, scenario, iter):
    starts, goals = get_starts_goals(config, rng, scenario)
    agents = dict()
    other_policy = ego_policy if config.env.homogeneous else config.env.human_policy
    policies = [ego_policy] + (config.env.num_of_agents - 1) * [other_policy]
    ids = range(iter * len(policies), iter * len(policies) + len(policies))
    for id, start, goal, policy in zip(ids, starts, goals, policies):
        module = importlib.import_module(f"policies.{policy}")
        cls = getattr(module, "".join(wd.capitalize() for wd in policy.split("_")))
        agents[id] = cls(config, env, id, policy, start, goal)
    return agents[ids[0]], agents


def get_starts_goals(config, rng, scenario):
    if scenario == "swap":
        starts = [[-config.env.interaction_dist / 2, 0], [config.env.interaction_dist / 2, 0]]
        goals = starts[::-1]
    elif scenario == "passing":
        starts = [
            [-config.env.interaction_dist / 2, 0],
            [config.env.interaction_dist / 2, 2 * config.agent.radius + config.agent.lat_dist],
        ]
        goals = [
            [config.env.interaction_dist / 2, 0],
            [-config.env.interaction_dist / 2, 2 * config.agent.radius + config.env.lat_dist],
        ]
    elif scenario == "acute":
        starts = [
            [-config.env.interaction_dist / 2, 0],
            [-config.env.interaction_dist / 2, -1.5],
        ]
        goals = [
            [config.env.interaction_dist / 2, 0],
            [config.env.interaction_dist / 2 - 2, 1.5],
        ]
    elif scenario == "obtuse":
        starts = [
            [-config.env.interaction_dist / 2, 0],
            [config.env.interaction_dist / 2 + 1, 2],
        ]
        goals = [
            [config.env.interaction_dist / 2, 0],
            [-config.env.interaction_dist / 2 + 1, -2],
        ]
    elif scenario == "2_agent_split":
        starts = [
            [-config.env.interaction_dist / 2, 0],
            [config.env.interaction_dist / 2, 2 * config.agent.radius + config.env.lat_dist],
            [config.env.interaction_dist / 2, -(2 * config.agent.radius + config.env.lat_dist)],
        ]
        goals = [
            [config.env.interaction_dist / 2, 0],
            [-config.env.interaction_dist / 2, 2 * config.agent.radius + config.env.lat_dist],
            [-config.env.interaction_dist / 2, -(2 * config.agent.radius + config.env.lat_dist)],
        ]
    elif scenario == "t_junction":
        starts = [
            [-config.env.interaction_dist / 2, 0],
            [0, -config.env.interaction_dist / 2],
        ]
        goals = [[config.env.interaction_dist / 2, 0], [0, config.env.interaction_dist / 2]]
    elif scenario == "2_agent_t_junction":
        starts = [
            [-config.env.interaction_dist / 2, 0],
            [-0.5, config.env.interaction_dist / 2 - 1],
            [0.5 * config.agent.radius + config.env.lat_dist, -config.env.interaction_dist / 2],
        ]
        goals = [
            [config.env.interaction_dist / 2, 0],
            [-0.5, -config.env.interaction_dist / 2],
            [0.5 * config.agent.radius + config.env.lat_dist, config.env.interaction_dist / 2],
        ]
    elif scenario == "custom":
        starts = [start_goal[0] for start_goal in config.env.custom_pos]
        goals = [start_goal[1] for start_goal in config.env.custom_pos]
    elif scenario == "circle":
        thetas = np.linspace(
            0, 2 * np.pi - 2 * np.pi / config.env.num_of_agents, config.env.num_of_agents
        )
        starts = config.env.circle_radius * helper.unit_vec(thetas)
        goals = config.env.circle_radius * helper.unit_vec(thetas + np.pi)
    elif scenario == "random":
        min_dist = 2 * config.agent.radius + config.env.min_start_buffer
        for _ in range(100):
            starts = config.env.workspace_length * rng.random((config.env.num_of_agents, 2))
            goals = config.env.workspace_length * rng.random((config.env.num_of_agents, 2))
            if (
                is_feasible(starts, min_dist)
                and is_feasible(goals, min_dist)
                and far_enough(starts, goals)
            ):
                break
        else:
            raise AttemptsExceededError(100)
    else:
        raise ValueError(f"Scenario '{scenario}' is not recognized")

    # Without noise, RVO might not work properly
    bound = 0.01
    starts += rng.uniform(-bound, bound, np.shape(starts))
    goals += rng.uniform(-bound, bound, np.shape(goals))

    return starts, goals
