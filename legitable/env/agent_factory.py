import ast
import numpy as np
import importlib

from utils import helper

class AttemptsExceededError(Exception):
    def __init__(self, max_attempts, message=None):
        self.max_attempts = max_attempts
        self.default_msg = f"Couldn't create feasible configuration within {self.max_attempts} attempts"
        self.message = self.default_msg if message is None else message
        super().__init__(self.message)

def is_feasible(positions, min_dist):
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if helper.dist(positions[i], positions[j]) < min_dist:
                return False
    return True

def far_enough(starts, goals):
    return np.all(helper.dist(starts, goals) > 2)

def init_agents(config, env, ego_policy, iter):
    starts, goals = get_starts_goals(config)
    agents = dict()
    other_policy = ego_policy if config.homogeneous else config.human_policy
    policies = [ego_policy] + (config.num_of_agents - 1) * [other_policy]
    ids = range(iter * len(policies), iter * len(policies) + len(policies))
    for id, start, goal, policy in zip(ids, starts, goals, policies):
        module = importlib.import_module(f'policies.{policy}')
        cls = getattr(module, ''.join(wd.capitalize() for wd in policy.split('_')))
        agents[id] = cls(config, env, id, policy, start, goal)
    return agents[ids[0]], agents

def get_starts_goals(config):
    if config.scenario == 'circle':
        thetas = np.linspace(0, 2*np.pi-2*np.pi/config.num_of_agents, config.num_of_agents)
        starts = config.circle_radius * helper.unit_vec(thetas)
        goals = config.circle_radius * helper.unit_vec(thetas + np.pi)
    elif config.scenario == 'random':
        min_dist = 2 * config.radius + config.min_start_buffer
        for _ in range(100):
            starts = config.workspace_length * np.random.rand(config.num_of_agents, 2)
            goals = config.workspace_length * np.random.rand(config.num_of_agents, 2)
            if is_feasible(starts, min_dist) and is_feasible(goals, min_dist) and far_enough(starts, goals):
                break
        else:
            raise AttemptsExceededError(100)
    elif config.scenario == 'swap':
        starts = [[-config.interaction_dist / 2, 0], [config.interaction_dist / 2, 0]]
        goals = starts[::-1]
    elif config.scenario == 'passing':
        starts = [[-config.interaction_dist / 2, 0], [config.interaction_dist / 2, 2 * config.radius + config.lat_dist]]
        goals = [[config.interaction_dist / 2, 0], [-config.interaction_dist / 2, 2 * config.radius + config.lat_dist]]
    elif config.scenario == '2_agent_split':
        starts = [[-config.interaction_dist / 2, 0], [config.interaction_dist / 2, 2 * config.radius + config.lat_dist], [config.interaction_dist / 2, -(2 * config.radius + config.lat_dist)]]
        goals = [[config.interaction_dist / 2, 0], [-config.interaction_dist / 2, 2 * config.radius + config.lat_dist], [-config.interaction_dist / 2, -(2 * config.radius + config.lat_dist)]]
    elif config.scenario == 't_junction':
        starts = [[-config.interaction_dist / 2, 0], [0, -config.interaction_dist / 2]]
        goals = [[config.interaction_dist / 2, 0], [0, config.interaction_dist / 2]]
    elif config.scenario == '2_agent_t_junction':
        starts = [[-config.interaction_dist / 2, 0], [0, config.interaction_dist / 2-1], [2 * config.radius + config.lat_dist, -config.interaction_dist / 2]]
        goals = [[config.interaction_dist / 2, 0], [0, -config.interaction_dist / 2], [2 * config.radius + config.lat_dist, config.interaction_dist / 2]]
    elif config.scenario == 'custom':
        starts = [start_goal[0] for start_goal in config.custom_pos]
        goals = [start_goal[1] for start_goal in config.custom_pos]
    else:
        raise ValueError(f"Scenario '{config.scenario}' is not recognized")

    # Without noise, RVO might not work properly
    bound = 0.01
    starts += np.random.uniform(-bound, bound, np.shape(starts))
    goals += np.random.uniform(-bound, bound, np.shape(goals))
    
    return starts, goals
