from policies.agent import Agent
from utils import helper
import numpy as np
import rvo2

class Rvo(Agent):

    def __init__(self, config, env, id, policy, start, goal):
        super().__init__(config, env, id, policy, start, goal)
        self.color = "#DC267F"
        # self.color = "#EC5F67"
        self.rvo_sim = rvo2.PyRVOSimulator(self.config.timestep, self.config.rvo_neighbor_dist, \
                self.config.rvo_max_neighbors, self.config.rvo_time_horiz, self.config.rvo_time_horiz, \
                self.config.radius, self.config.max_speed)

    def post_init(self):
        super().post_init()
        self.rvo_agents = {id : self.rvo_sim.addAgent(tuple(a.start)) for id, a in self.env.agents.items()}

    def get_action(self):
        for id, agent in self.rvo_agents.items():
            self.rvo_sim.setAgentPosition(agent, tuple(self.env.agents[id].pos))
            pref_vel = self.env.agents[id].max_speed * helper.unit_vec(helper.angle(self.env.agents[id].goal \
                    - self.env.agents[id].pos))
            self.rvo_sim.setAgentPrefVelocity(agent, tuple(pref_vel))
        self.rvo_sim.doStep()

        dpos = self.rvo_sim.getAgentPosition(self.rvo_agents[self.id]) - self.pos
        self.des_speed = np.linalg.norm(dpos) / self.config.timestep
        self.des_heading = helper.angle(dpos)
