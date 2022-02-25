import numpy as np
import rvo2
from policies.agent import Agent
from utils import helper


class Rvo(Agent):
    def __init__(self, config, env, id, policy, is_ego, start, goal=None, max_speed=None):
        super().__init__(config, env, id, policy, is_ego, start, goal=goal, max_speed=max_speed)
        self.color = "#868293"
        self.neighbor_dist = self.conf.neighbor_dist
        self.max_neighbors = self.conf.max_neighbors
        self.time_horiz = self.conf.time_horiz
        self.rvo_sim = rvo2.PyRVOSimulator(
            self.env.dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horiz,
            self.time_horiz,
            self.radius * self.conf.scaled_radius,
            self.max_speed,
        )

    def post_init(self):
        super().post_init()
        self.rvo_agents = {
            id: self.rvo_sim.addAgent(tuple(a.start)) for id, a in self.env.agents.items()
        }

    def get_action(self):
        for id, agent in self.rvo_agents.items():
            self.rvo_sim.setAgentPosition(agent, tuple(self.env.agents[id].pos))
            self.rvo_sim.setAgentVelocity(agent, tuple(self.env.agents[id].vel))
        pref_vel = self.max_speed * helper.unit_vec(self.goal - self.pos)
        self.rvo_sim.setAgentPrefVelocity(self.rvo_agents[self.id], tuple(pref_vel))
        self.rvo_sim.doStep()

        dpos = self.rvo_sim.getAgentPosition(self.rvo_agents[self.id]) - self.pos
        self.des_speed = np.linalg.norm(dpos) / self.env.dt
        self.des_heading = helper.angle(dpos)
