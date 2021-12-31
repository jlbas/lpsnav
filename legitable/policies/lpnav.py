import numpy as np
from policies.agent import Agent
from utils import helper
from utils.animation import snapshot


class Lpnav(Agent):
    def __init__(self, config, env, id, policy, start, goal=None, max_speed=None):
        super().__init__(config, env, id, policy, start, goal=goal, max_speed=max_speed)
        self.receding_horiz = self.conf.receding_horiz
        self.sensing_horiz = self.conf.sensing_horiz
        self.receding_steps = int(self.receding_horiz / self.env.dt)
        self.speed_samples = self.conf.speed_samples
        self.heading_samples = self.conf.heading_samples
        self.subgoal_priors = np.array(self.conf.subgoal_priors)
        self.leg_tol = self.conf.legibility_tol
        self.beta = self.conf.beta
        self.color = "#785EF0"
        self.color = "#774db9"
        self.pred_pos = dict()
        self.int_lines = dict()
        self.pred_int_lines = dict()
        self.interacting_agents = dict()
        self.cost_rt = self.receding_horiz
        self.cost_tp = self.prim_horiz
        self.cost_tg = dict()
        self.cost_pg = dict()
        self.cost_rg = dict()
        self.cost_tpg = dict()
        self.cost_rpg = dict()
        self.cost_rtg = dict()
        self.prim_leg_score = dict()
        self.prim_pred_score = dict()
        self.current_leg_score = dict()
        self.passing_ratio = dict()
        self.is_legible = dict()
        self.abs_prims_log = np.full(
            (
                int(self.env.max_duration / self.env.dt) + 1,
                self.speed_samples,
                self.heading_samples,
                2,
            ),
            np.inf,
        )
        self.opt_log = list()
        self.col_mask_log = list()
        self.abs_prim_vels = np.multiply.outer(self.speeds, helper.vec(self.abs_headings))
        self.speed_idx = 0
        self.heading_idx = self.heading_samples // 2
        self.col_mask = np.full((self.speed_samples, self.heading_samples), False)

    def post_init(self):
        super().post_init()
        self.taus = {id: 0 for id in self.other_agents}
        self.int_start_t = {id: -1 for id in self.other_agents}
        self.int_t = {id: -1 for id in self.other_agents}
        self.int_lines_log = {
            id: np.full((int(self.env.max_duration / self.env.dt) + 1, 2, 2), np.inf)
            for id in self.other_agents
        }
        self.pred_int_lines_log = {
            id: np.full((int(self.env.max_duration / self.env.dt) + 1, 2, 2), np.inf)
            for id in self.other_agents
        }
        self.col_circle_log = {
            id: np.full((int(self.env.max_duration / self.env.dt) + 1, 2), np.inf)
            for id in self.other_agents
        }
        col_width = {
            id: self.radius + a.radius + self.conf.col_buffer for id, a in self.other_agents.items()
        }
        self.rel_int_line = {id: np.array([[0, -cw], [0, cw]]) for id, cw in col_width.items()}

    def update_abs_prim_vels(self):
        self.abs_prim_vels = np.multiply.outer(self.speeds, helper.vec(self.abs_headings))

    def update_int_line(self):
        self.int_line_heading = helper.wrap_to_pi(helper.angle(self.pos - self.goal))
        self.rel_int_lines = dict()
        for id, agent in self.other_agents.items():
            self.rel_int_lines[id] = helper.rotate(self.rel_int_line[id], self.int_line_heading)
            self.int_lines[id] = self.rel_int_lines[id] + agent.pos

    def get_interacting_agents(self):
        self.interacting_agents = dict()
        for id, agent in self.other_agents.items():
            time_to_interaction = helper.cost_to_line(
                self.pos, self.speed, self.int_lines[id], agent.vel
            )
            in_front = helper.in_front(agent.pos, self.int_line_heading, self.pos)
            if in_front and time_to_interaction < self.sensing_horiz:
                self.interacting_agents[id] = agent

    def remove_col_prims(self):
        self.col_mask = np.full((self.speed_samples, self.heading_samples), False)
        for t in np.linspace(0, 1, 10):
            ego_pred = self.pos + t * (self.abs_prims - self.pos)
            for a in self.other_agents.values():
                a_pred = a.pos + t * (a.pos + a.vel * self.prim_horiz - a.pos)
                self.col_mask |= helper.dist(ego_pred, a_pred) < 2 * self.radius + self.radius / 2

    def predict_pos(self, id, agent):
        self.pred_pos[id] = agent.pos + agent.vel * self.prim_horiz
        self.pred_int_lines[id] = self.pred_pos[id] + self.rel_int_lines[id]

    def update_int_t(self, id):
        if id in self.interacting_agents:
            if self.int_start_t[id] == -1:
                self.int_start_t[id] = self.env.time
            self.int_t[id] = self.env.time - self.int_start_t[id]
        else:
            self.int_start_t[id] = -1

    def get_int_costs(self, id, agent):
        if self.env.time < self.receding_horiz:
            receded_pos = self.pos_log[0] - self.vel_log[0] * (self.receding_horiz - self.env.time)
        else:
            receded_pos = self.pos_log[self.env.step - self.receding_steps]
        receded_line = self.int_lines[id] - agent.vel * self.receding_horiz
        self.cost_rg[id] = helper.dynamic_pt_cost(
            receded_pos,
            self.max_speed,
            receded_line,
            self.int_line_heading,
            agent.vel,
        )
        self.cost_tg[id] = helper.dynamic_pt_cost(
            self.pos,
            self.max_speed,
            self.int_lines[id],
            self.int_line_heading,
            agent.vel,
        )
        self.cost_pg[id] = helper.dynamic_prim_cost(
            self.pos,
            self.abs_prims,
            self.max_speed,
            self.abs_prim_vels,
            self.pred_int_lines[id],
            self.int_line_heading,
            agent.vel,
            self.int_lines[id],
        )
        self.cost_tpg[id] = self.cost_tp + self.cost_pg[id]
        if np.any(self.cost_pg[id] == 0):
            partial_cost_tpg = helper.directed_cost_to_line(
                self.pos, self.abs_prim_vels, self.int_lines[id], agent.vel
            )
            self.cost_tpg[id] = np.where(self.cost_pg[id] == 0, partial_cost_tpg, self.cost_tpg[id])

    def compute_prim_leg(self, id):
        arg = self.cost_sg[id][..., None, None] - self.cost_spg[id]
        with np.errstate(under="ignore"):
            self.prim_leg_score[id] = np.exp(arg) * self.subgoal_priors[..., None, None]
            self.prim_leg_score[id] = np.around(self.prim_leg_score[id], 10)
        self.prim_leg_score[id] = self.prim_leg_score[id] / np.sum(self.prim_leg_score[id], axis=0)
        self.prim_leg_score[id] = np.delete(self.prim_leg_score[id], 1, 0)
        assert np.all(np.around(self.prim_leg_score[id], 5) <= 1), "Error in legibility computation"
        self.cost_rpg[id] = self.cost_rt + self.cost_tpg[id]
        self.cost_rtg[id] = self.cost_rt + self.cost_tg[id]

    def compute_leg(self, id):
        arg = self.cost_rg[id] - self.cost_rtg[id]
        with np.errstate(under="ignore"):
            self.current_leg_score[id] = np.exp(arg) * self.subgoal_priors
        self.current_leg_score[id][
            np.isclose(self.current_leg_score[id], 0, atol=self.env.epsilon)
        ] = self.env.epsilon
        self.current_leg_score[id] /= np.sum(self.current_leg_score[id])
        self.current_leg_score[id] = np.delete(self.current_leg_score[id], 1)
        assert np.all(
            np.around(self.current_leg_score[id], 5) <= 1
        ), "Error in current legibility computation"

    def compute_prim_pred(self, id, agent):
        arg = self.cost_tg[id][..., None, None] - self.cost_tpg[id]
        arg = np.delete(arg, 1, 0)[np.argmax(self.current_leg_score[id])]
        with np.errstate(under="ignore"):
            self.prim_pred_score[id] = np.exp(arg)
            self.prim_pred_score[id] = np.around(self.prim_pred_score[id], 10)
        assert np.all(
            np.around(self.prim_pred_score[id], 5) <= 1
        ), "Error in predictability computation"

    def check_if_legible(self, id):
        self.passing_ratio[id] = np.max(self.current_leg_score[id]) / max(
            np.min(self.current_leg_score[id]), self.env_conf.epsilon
        )
        self.is_legible[id] = self.passing_ratio[id] > self.leg_tol

    def update_tau(self, id, agent):
        if self.is_legible[id] and (not self.int_t[id] or self.taus[id] == 1) or agent.speed == 0:
            self.taus[id] = 1
        else:
            x = min(self.passing_ratio - 1, 10)
            self.taus[id] = -1 + 2 / (1 + np.exp(-self.beta * x))

    def get_leg_pred_prims(self):
        score = np.full((self.speed_samples, self.heading_samples), np.inf)
        for id in self.interacting_agents:
            new_score = (1 - self.taus[id]) * self.prim_leg_score[id] + self.taus[
                id
            ] * self.prim_pred_score[id]
            score = np.minimum(score, np.max(new_score, axis=0))
        score = np.where(self.col_mask, -np.inf, score)
        is_max = score == np.max(score)
        if np.sum(is_max) > 1:
            self.get_goal_prims(~is_max)
        else:
            self.speed_idx, self.heading_idx = np.unravel_index(np.argmax(score), score.shape)

    def get_goal_prims(self, mask=None):
        delta_heading = np.abs(self.abs_headings - helper.angle(self.goal - self.pos))
        goal_cost = np.tile(delta_heading, (self.speed_samples, 1))
        inf_mask = self.col_mask if mask is None else self.col_mask | mask
        goal_cost = np.where(inf_mask, np.inf, goal_cost)
        self.speed_idx, self.heading_idx = np.unravel_index(np.argmin(goal_cost), goal_cost.shape)

    def get_action(self):
        self.update_abs_prims()
        self.update_abs_headings()
        self.update_abs_prim_vels()
        self.update_int_line()
        self.get_interacting_agents()
        self.remove_col_prims()
        for id, agent in self.interacting_agents.items():
            self.predict_pos(id, agent)
            self.update_int_t(id)
            self.get_int_costs(id, agent)
            self.compute_prim_leg(id)
            self.compute_leg(id)
            self.compute_prim_pred(id, agent)
            self.check_if_legible(id)
            self.update_tau(id, agent)
        if np.all(self.col_mask):
            self.des_speed = 0
            self.des_heading = self.heading
        else:
            if self.interacting_agents:
                self.get_leg_pred_prims()
            else:
                self.get_goal_prims()
            self.des_speed = self.speeds[self.speed_idx]
            self.des_heading = self.abs_headings[self.heading_idx]

    def log_data(self, step):
        super().log_data(step)
        self.abs_prims_log[step] = self.abs_prims
        self.opt_log.append([self.speed_idx, self.heading_idx])
        self.col_mask_log.append(self.col_mask)
        for id in self.other_agents:
            if id in self.interacting_agents:
                self.int_lines_log[id][step] = self.int_lines[id]
                self.pred_int_lines_log[id][step] = self.pred_int_lines[id]
                self.col_circle_log[id][step] = agent.pos
            else:
                self.int_lines_log[id][step] = 2 * [None]
                self.pred_int_lines_log[id][step] = 2 * [None]
                self.col_circle_log[id][step] = None
