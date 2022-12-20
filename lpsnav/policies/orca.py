import numpy as np
import rvo2
from policies.agent import Agent
from utils import helper


class Orca(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.neighbor_dist = conf["neighbor_dist"]
        self.max_neighbors = conf["max_neighbors"]
        self.time_horiz = conf["time_horiz"]
        self.expanded_radius = self.radius * conf["scaled_radius"]

    def post_init(self, dt, agents):
        super().post_init(dt, agents)
        self.orca_sim = rvo2.PyRVOSimulator(
            dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horiz,
            self.time_horiz,
            self.expanded_radius,
            self.max_speed,
        )
        self.orca_sim.processObstacles()
        self.orca_agents = {self.id: self.orca_sim.addAgent(tuple(self.pos))}
        for k, a in agents.items():
            self.orca_agents[k] = self.orca_sim.addAgent(tuple(a.pos))

    def get_action(self, dt, agents):
        self.orca_sim.setAgentPosition(self.orca_agents[self.id], tuple(self.pos))
        self.orca_sim.setAgentVelocity(self.orca_agents[self.id], tuple(self.vel))
        for k, a in agents.items():
            self.orca_sim.setAgentPosition(self.orca_agents[k], tuple(a.pos))
            self.orca_sim.setAgentVelocity(self.orca_agents[k], tuple(a.vel))
        pref_vel = self.max_speed * helper.unit_vec(self.goal - self.pos)
        self.orca_sim.setAgentPrefVelocity(self.orca_agents[self.id], tuple(pref_vel))
        self.orca_sim.doStep()

        dpos = self.orca_sim.getAgentPosition(self.orca_agents[self.id]) - self.pos
        self.des_speed = np.linalg.norm(dpos) / dt
        self.des_heading = helper.angle(dpos)
