import numpy as np
from policies.agent import Agent
from utils import helper


class SocialMomentum(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.sm_weight = conf["weight"]
        self.pred_pos = {}
        self.agent_weights = {}
        self.sm_cost = {}

    def update_prim_vels(self):
        self.prim_vels = np.multiply.outer(self.speeds, helper.vec(self.abs_headings))

    def get_interacting_agents(self, agents):
        self.interacting_agents = {}
        for k, a in agents.items():
            if helper.in_front(self.pos, self.heading, a.pos):
                self.interacting_agents[k] = a

    def get_goal_score(self, dt):
        next_pos = self.pos + dt * self.abs_prim_vels
        self.goal_score = 1 / helper.dist(next_pos, self.goal)

    def predict_pos(self, id, agent):
        self.pred_pos[id] = agent.pos + agent.vel * self.prim_horiz

    def get_agent_weights(self):
        for k, agent in self.interacting_agents.items():
            self.agent_weights[k] = 1 / helper.dist(self.pos, agent.pos)
        normalizer = sum(self.agent_weights.values())
        for k in self.interacting_agents:
            self.agent_weights[k] /= normalizer

    def get_sm_cost(self, id, agent):
        r_c = (self.pos + agent.pos) / 2
        r_ac = self.pos - r_c
        r_bc = agent.pos - r_c
        l_ab = np.cross(r_ac, self.vel) + np.cross(r_bc, agent.vel)
        r_c_hat = (self.abs_prims + self.pred_pos[id]) / 2
        r_ac_hat = self.abs_prims - r_c_hat
        r_bc_hat = self.pred_pos[id] - r_c_hat
        l_ab_hat = np.cross(r_ac_hat, self.prim_vels) + np.cross(r_bc_hat, agent.vel)
        self.sm_cost[id] = np.where(
            np.dot(l_ab, l_ab_hat) > 0, self.agent_weights[id] * np.abs(l_ab_hat), 0
        )

    def get_action(self, dt, agents):
        self.update_abs_prims()
        self.update_abs_headings()
        self.update_prim_vels()
        self.update_abs_prim_vels()
        self.remove_col_prims(dt, agents)
        self.get_interacting_agents(agents)
        self.get_goal_score(dt)
        self.get_agent_weights()
        for k, agent in self.interacting_agents.items():
            self.predict_pos(k, agent)
            self.get_sm_cost(k, agent)
        if self.interacting_agents:
            sm_score = np.zeros((self.speed_samples, self.heading_samples))
            for k in self.interacting_agents:
                sm_score += self.sm_cost[k]
            self.goal_score /= np.max(self.goal_score)
            with np.errstate(divide="ignore", invalid="ignore"):
                sm_score = np.nan_to_num(sm_score / np.max(sm_score))
            score = self.goal_score + self.sm_weight * sm_score
        else:
            score = self.goal_score.copy()
        score = np.where(self.col_mask, -np.inf, score)
        is_max = score == np.max(score)
        if np.sum(is_max) > 1:
            score = np.where(is_max, self.goal_score, -np.inf)
        speed_idx, heading_idx = np.unravel_index(np.argmax(score), score.shape)
        self.des_speed = self.speeds[speed_idx]
        self.des_heading = self.abs_headings[heading_idx]
