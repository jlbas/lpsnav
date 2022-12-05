import numpy as np
import rvo2
from policies.agent import Agent
from utils import helper


class Rvo(Agent):
        self.neighbor_dist = conf["neighbor_dist"]
        self.max_neighbors = conf["max_neighbors"]
        self.time_horiz = conf["time_horiz"]
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.expanded_radius = self.radius * conf["scaled_radius"]

    def post_init(self, dt, agents):
        super().post_init(dt, agents)
        self.rvo_sim = rvo2.PyRVOSimulator(
            dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horiz,
            self.time_horiz,
            self.expanded_radius,
            self.max_speed,
        )
        self.rvo_sim.processObstacles()
        self.rvo_agents = {self.id: self.rvo_sim.addAgent(tuple(self.pos))}
        for k, a in agents.items():
            self.rvo_agents[k] = self.rvo_sim.addAgent(tuple(a.pos))

    def get_action(self, dt, agents):
        self.rvo_sim.setAgentPosition(self.rvo_agents[self.id], tuple(self.pos))
        self.rvo_sim.setAgentVelocity(self.rvo_agents[self.id], tuple(self.vel))
        for k, a in agents.items():
            self.rvo_sim.setAgentPosition(self.rvo_agents[k], tuple(a.pos))
            self.rvo_sim.setAgentVelocity(self.rvo_agents[k], tuple(a.vel))
        pref_vel = self.max_speed * helper.unit_vec(self.goal - self.pos)
        self.rvo_sim.setAgentPrefVelocity(self.rvo_agents[self.id], tuple(pref_vel))
        self.rvo_sim.doStep()

        dpos = self.rvo_sim.getAgentPosition(self.rvo_agents[self.id]) - self.pos
        self.des_speed = np.linalg.norm(dpos) / dt
        self.des_heading = helper.angle(dpos)
