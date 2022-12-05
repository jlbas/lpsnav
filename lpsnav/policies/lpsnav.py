import numpy as np
from policies.agent import Agent
from utils import helper


class Lpsnav(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.heading_samples = conf["heading_samples"]
        self.leg_tol = conf["legibility_tol"]
        self.pred_tol = conf["predictability_tol"]
        self.max_cost = conf["max_cost"]
        self.receding_horiz = conf["receding_horiz"]
        self.sensing_horiz = conf["sensing_horiz"]
        self.speed_samples = conf["speed_samples"]
        self.long_space = conf["longitudinal_space"]
        self.lat_space = conf["lateral_space"]
        self.subgoal_priors = np.array(conf["subgoal_priors"])
        self.pred_pos = {}
        self.int_lines = {}
        self.pred_int_lines = {}
        self.cost_rt = self.receding_horiz
        self.cost_tp = self.prim_horiz
        self.cost_tg = {}
        self.cost_pg = {}
        self.cost_rg = {}
        self.cost_tpg = {}
        self.cost_rpg = {}
        self.cost_rtg = {}
        self.prim_leg_score = {}
        self.prim_pred_score = {}
        self.current_leg_score = {}
        self.passing_ratio = {}
        self.speed_idx = 0
        self.heading_idx = self.heading_samples // 2
        self.col_mask = np.full((self.speed_samples, self.heading_samples), False)
        self.tau = {}
        self.pass_inf_diff_hist = {}

    def post_init(self, dt, agents):
        super().post_init(dt, agents)
        t_hist = np.linspace(dt, self.receding_horiz, int(self.receding_horiz / dt))
        self.pos_hist = self.pos - self.vel * t_hist[:, None]
        self.int_start = {k: -1 for k in agents}

    def update_int_line(self, agents):
        self.int_line_heading = helper.wrap_to_pi(helper.angle(self.pos - self.goal))
        self.rel_int_lines = {}
        n = len(agents)
        for k, a in agents.items():
            dtheta = helper.wrap_to_pi(a.heading - self.int_line_heading - np.pi / 2)
            lat_space = self.lat_space[0] + (self.lat_space[1] - self.lat_space[0]) / n
            long_space = self.long_space[0] + (self.long_space[1] - self.long_space[0]) / n
            long_space = lat_space + min(1, a.speed / self.max_speed) * (long_space - lat_space)
            r = self.radius + a.radius
            col_width = helper.polar_ellipse(long_space + r, lat_space + r, dtheta)
            self.rel_int_lines[k] = helper.rotate(np.array([[0, -col_width], [0, col_width]]), self.int_line_heading)
            self.int_lines[k] = self.rel_int_lines[k] + a.pos

    def get_interacting_agents(self, agents):
        self.interacting_agents = {}
        for k, a in agents.items():
            time_to_interaction = helper.cost_to_line(
                self.pos, self.speed, self.int_lines[k], a.vel
            )
            in_radius = helper.dist(self.pos, a.pos) < self.sensing_dist
            in_horiz = time_to_interaction < self.sensing_horiz
            intersecting = helper.is_intersecting(self.pos, self.goal, *self.int_lines[k])
            if in_radius and in_horiz and intersecting:
                self.interacting_agents[k] = a

    def predict_pos(self, id, agent):
        self.pred_pos[id] = agent.pos + agent.vel * self.prim_horiz
        self.pred_int_lines[id] = self.pred_pos[id] + self.rel_int_lines[id]

    def update_int_start(self, k):
        if k in self.interacting_agents:
            self.int_start[k] = 1 if self.int_start[k] == -1 else 0
        else:
            self.int_start[k] = -1

    def get_int_costs(self, id, agent):
        receded_line = self.int_lines[id] - agent.vel * self.receding_horiz
        self.cost_rg[id] = helper.dynamic_pt_cost(
            self.pos_hist[-1],
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
        self.cost_rpg[id] = self.cost_rt + self.cost_tpg[id]
        self.cost_rtg[id] = self.cost_rt + self.cost_tg[id]

    def compute_leg(self, id):
        arg = self.cost_rg[id] - self.cost_rtg[id]
        arg = np.where(self.cost_rtg[id] > self.max_cost, -np.inf, arg)
        arg = np.clip(arg, -self.max_cost, 0)
        self.current_leg_score[id] = np.exp(arg) * self.subgoal_priors
        self.current_leg_score[id] /= np.sum(self.current_leg_score[id])
        self.current_leg_score[id] = np.delete(self.current_leg_score[id], 1)

    def compute_prim_leg(self, id):
        arg = self.cost_rg[id][..., None, None] - self.cost_rpg[id]
        arg = np.where(self.cost_rpg[id] > self.max_cost, -np.inf, arg)
        arg = np.clip(arg, -self.max_cost, 0)
        self.prim_leg_score[id] = np.exp(arg) * self.subgoal_priors[..., None, None]
        self.prim_leg_score[id] = self.prim_leg_score[id] / np.sum(self.prim_leg_score[id], axis=0)
        self.prim_leg_score[id] = np.delete(self.prim_leg_score[id], 1, 0)

    def compute_prim_pred(self, id, agent):
        arg = self.cost_tg[id][..., None, None] - self.cost_tpg[id]
        arg = np.where(self.cost_tpg[id] > self.max_cost, -np.inf, arg)
        arg = np.clip(arg, -self.max_cost, 0)
        arg = np.delete(arg, 1, 0)
        if agent.speed != 0:
            arg = arg[np.argmax(self.current_leg_score[id])]
        else:
            arg = arg[np.argmin(helper.dist(self.goal, self.int_lines[id]))]
        self.prim_pred_score[id] = np.exp(arg)

    def update_passing_ratio(self, id):
        pass_inf_diff = np.max(self.current_leg_score[id]) - np.min(self.current_leg_score[id])
        if id not in self.pass_inf_diff_hist:
            self.pass_inf_diff_hist[id] = np.full(len(self.pos_hist), pass_inf_diff)
        else:
            self.pass_inf_diff_hist[id] = np.roll(self.pass_inf_diff_hist[id], 1, axis=0)
            self.pass_inf_diff_hist[id][0] = pass_inf_diff
        self.passing_ratio[id] = np.mean(self.pass_inf_diff_hist[id])

    def update_tau(self, id):
        num = self.passing_ratio[id] - self.leg_tol
        den = self.pred_tol - self.leg_tol
        self.tau[id] = max(0, min(1, num / den))

    def get_leg_pred_prims(self, dt):
        score = np.full((self.speed_samples, self.heading_samples), np.inf)
        for k in self.interacting_agents:
            new_score = (1 - self.tau[k]) * self.prim_leg_score[k] + self.tau[
                k
            ] * self.prim_pred_score[k]
            score = np.minimum(score, np.max(new_score, axis=0))
        score = np.where(self.col_mask, -np.inf, score)
        is_max = score == np.max(score)
        if np.sum(is_max) > 1:
            self.get_goal_prims(dt, ~is_max)
        else:
            self.speed_idx, self.heading_idx = np.unravel_index(np.argmax(score), score.shape)

    def get_goal_prims(self, dt, mask=None):
        next_pos = self.pos + dt * self.abs_prim_vels
        goal_cost = helper.dist(next_pos, self.goal)
        inf_mask = self.col_mask if mask is None else self.col_mask | mask
        goal_cost = np.where(inf_mask, np.inf, goal_cost)
        self.speed_idx, self.heading_idx = np.unravel_index(np.argmin(goal_cost), goal_cost.shape)

    def get_action(self, dt, agents):
        self.update_abs_prims()
        self.update_abs_headings()
        self.update_abs_prim_vels()
        self.update_int_line(agents)
        self.get_interacting_agents(agents)
        for k, a in agents.items():
            if k in self.interacting_agents:
                self.predict_pos(k, a)
                self.update_int_start(k)
                self.get_int_costs(k, a)
                self.compute_leg(k)
                self.compute_prim_leg(k)
                self.compute_prim_pred(k, a)
                self.update_passing_ratio(k)
                self.update_tau(k)
            else:
                self.pass_inf_diff_hist.pop(k, None)
        self.remove_col_prims(dt, agents)
        if np.all(self.col_mask):
            self.des_speed = self.min_speed
            self.des_heading = self.heading
        else:
            if self.interacting_agents:
                self.get_leg_pred_prims(dt)
            else:
                self.get_goal_prims(dt)
            self.des_speed = self.speeds[self.speed_idx]
            self.des_heading = self.abs_headings[self.heading_idx]
        self.pos_hist = np.roll(self.pos_hist, 1, axis=0)
        self.pos_hist[0] = self.pos
