import numpy as np
from policies.agent import Agent
from utils import helper
from utils.animation import snapshot


class Legitable(Agent):
    def __init__(self, config, env, id, policy, start, goal):
        super().__init__(config, env, id, policy, start, goal)
        self.scaled_speed = self.config.scaled_speed
        self.sensing_dist = self.config.sensing_dist
        self.receding_horiz = self.config.receding_horiz
        self.receding_steps = int(self.receding_horiz / self.env.timestep)
        self.speed_samples = self.config.speed_samples
        self.heading_samples = self.config.heading_samples
        col_width = 2 * self.radius + self.config.col_buffer
        self.int_baseline = np.array([[0, -col_width], [0, col_width]])
        self.subgoal_priors = np.array(config.subgoal_priors)
        self.leg_tol = self.config.legibility_tol
        self.color = "#785EF0"
        # self.color = "#C594c5"
        self.pred_pos = dict()
        self.int_lines = dict()
        self.pred_int_lines = dict()
        self.interacting_agents = dict()
        self.cost_tp = dict()
        self.cost_tg = dict()
        self.cost_pg = dict()
        self.cost_st = dict()
        self.cost_sg = dict()
        self.cost_tpg = dict()
        self.cost_spg = dict()
        self.prim_leg_score = dict()
        self.prim_pred_score = dict()
        self.current_leg_score = dict()
        self.pareto_front = dict()
        self.is_legible = dict()
        self.abs_prims_log = np.full(
            (
                int(self.env.max_duration / self.env.timestep) + 1,
                self.speed_samples,
                self.heading_samples,
                2,
            ),
            np.inf,
        )
        self.opt_log = list()
        self.col_mask_log = list()
        self.abs_prim_vels = np.multiply.outer(
            self.speeds, helper.unit_vec(self.abs_headings)
        )
        self.speed_idx = 0
        self.heading_idx = self.heading_samples // 2
        self.col_mask = np.full((self.speed_samples, self.heading_samples), False)

    def post_init(self):
        super().post_init()
        self.taus = {id: 0 for id in self.other_agents}
        self.pred_int_t = {id: -1 for id in self.other_agents}
        self.int_start_t = {id: -1 for id in self.other_agents}
        self.int_t = {id: -1 for id in self.other_agents}
        self.int_lines_log = {
            id: np.full(
                (int(self.env.max_duration / self.env.timestep) + 1, 2, 2), np.inf
            )
            for id in self.other_agents
        }
        self.pred_int_lines_log = {
            id: np.full(
                (int(self.env.max_duration / self.env.timestep) + 1, 2, 2), np.inf
            )
            for id in self.other_agents
        }

    def update_abs_prim_vels(self):
        self.abs_prim_vels = np.multiply.outer(
            self.speeds, helper.unit_vec(self.abs_headings)
        )

    def update_int_line(self):
        # if not hasattr(self, 'int_line_heading'):
        #     self.int_line_heading = helper.wrap_to_pi(np.pi + helper.angle(self.goal - self.pos))
        self.int_line_heading = helper.wrap_to_pi(
            np.pi + helper.angle(self.goal - self.pos)
        )
        self.int_pts = helper.rotate(self.int_baseline, self.int_line_heading)

    def get_interacting_agents(self):
        self.interacting_agents = dict()
        for id, agent in self.other_agents.items():
            self.int_lines[id] = agent.pos + self.int_pts
            in_front = helper.in_front(agent.pos, self.int_line_heading, self.pos)
            in_radius = helper.dist(self.pos, agent.pos) <= self.sensing_dist
            ttg = helper.dynamic_pt_cost(
                self.pos,
                self.max_speed,
                self.int_lines[id],
                self.int_line_heading,
                agent.vel,
            )
            approaching = np.min(np.delete(ttg, 1)) < 1e2
            # prims_in_front = np.all(helper.in_front(agent.pos, self.int_line_heading, self.abs_prims))
            if in_front and in_radius and approaching:
                self.interacting_agents[id] = agent

    def remove_col_prims(self):
        self.col_mask = np.full((self.speed_samples, self.heading_samples), False)
        for t in np.linspace(0, 1, 10):
            ego_pred = self.pos + t * (self.abs_prims - self.pos)
            for a in self.other_agents.values():
                a_pred = a.pos + t * (a.pos + a.vel * self.prim_horiz - a.pos)
                self.col_mask |= (
                    helper.dist(ego_pred, a_pred) < 2 * self.radius + self.radius
                )

    def predict_pos(self, id, agent):
        self.pred_pos[id] = agent.pos + agent.vel * self.prim_horiz
        self.pred_int_lines[id] = self.pred_pos[id] + self.int_pts
        self.cost_tg[id] = helper.dynamic_pt_cost(
            self.pos,
            self.scaled_speed,
            self.int_lines[id],
            self.int_line_heading,
            agent.vel,
        )

    def update_pred_int_t(self, id, agent):
        if id in self.interacting_agents:
            if self.int_start_t[id] == -1:
                ttg = helper.dynamic_pt_cost(
                    self.pos,
                    self.max_speed,
                    self.int_lines[id],
                    self.int_line_heading,
                    agent.vel,
                )
                self.pred_int_t[id] = np.min(np.delete(ttg, 1))
                self.int_start_t[id] = self.env.time
                self.int_t[id] = self.env.time
            else:
                self.int_t[id] = self.env.time - self.int_start_t[id]
        else:
            self.pred_int_t[id] = -1
            self.int_start_t[id] = -1

    def get_int_costs(self, id, agent):
        idx = max(0, self.env.step - self.receding_steps)
        receded_line = self.int_lines[id] - agent.vel * min(self.env.time, self.receding_horiz)
        self.cost_sg[id] = helper.dynamic_pt_cost(
            self.pos_log[idx],
            self.scaled_speed,
            receded_line,
            self.int_line_heading,
            agent.vel,
        )
        self.cost_pg[id] = helper.dynamic_prim_cost(
            self.pos,
            self.abs_prims,
            self.scaled_speed,
            self.abs_prim_vels,
            self.pred_int_lines[id],
            self.int_line_heading,
            agent.vel,
            self.int_lines[id],
        )
        self.cost_tpg[id] = self.prim_horiz + self.cost_pg[id]
        if np.any(self.cost_pg[id] == 0):
            partial_cost_tpg = helper.directed_cost_to_line(
                self.pos, self.abs_prim_vels, self.int_lines[id], agent.vel
            )
            self.cost_tpg[id] = np.where(
                self.cost_pg[id] == 0, partial_cost_tpg, self.cost_tpg[id]
            )
        self.cost_st[id] = min(self.env.time, self.receding_horiz)
        self.cost_spg[id] = self.cost_st[id] + self.cost_tpg[id]

    def compute_prim_leg(self, id):
        # snapshot(self, id)
        arg = self.cost_sg[id][..., None, None] - self.cost_spg[id]
        assert np.all(np.around(arg, 8) <= 0), "Error in legibility computation"
        bound = 2 * np.min(arg, where=np.isfinite(arg), initial=0)
        arg = np.nan_to_num(arg, nan=bound, posinf=bound, neginf=bound)
        self.prim_leg_score[id] = np.exp(arg) * self.subgoal_priors[..., None, None]
        self.prim_leg_score[id] /= np.sum(self.prim_leg_score[id], axis=0)
        self.prim_leg_score[id] = np.delete(self.prim_leg_score[id], 1, 0)
        assert np.all(
            np.around(self.prim_leg_score[id], 8) <= 1
        ), "Error in legibility computation"

    def compute_leg(self, id):
        # with np.errstate(invalid='ignore', over='ignore'):
        #     arg = self.cost_sg[id] - (self.cost_st[id] + self.cost_tg[id])
        #     bound = 2 * np.min(arg, where=np.isfinite(arg), initial=0)
        # arg = np.nan_to_num(arg, nan=bound, posinf=bound, neginf=bound)
        # self.current_leg_score[id] = np.exp(arg) * self.subgoal_priors
        # self.current_leg_score[id] /= np.sum(self.current_leg_score[id])
        # self.current_leg_score[id] = np.delete(self.current_leg_score[id], 1)
        speed_idx = np.argmin(np.abs(self.speeds - self.speed))
        heading_idx = self.heading_samples // 2
        self.current_leg_score[id] = self.prim_leg_score[id][:, speed_idx, heading_idx]

    def compute_prim_pred(self, id):
        arg = self.cost_tg[id][..., None, None] - self.cost_tpg[id]
        assert np.all(np.around(arg, 8) <= 0), "Error in predictability computation"
        bound = 2 * np.min(arg, where=np.isfinite(arg), initial=0)
        arg = np.nan_to_num(arg, nan=bound, posinf=bound, neginf=bound)
        # arg = np.where(self.prim_leg_score[id][0] > self.prim_leg_score[id][1], arg[0], arg[2])
        arg = np.delete(arg, 1, 0)[np.argmax(self.current_leg_score[id])]
        self.prim_pred_score[id] = np.exp(arg)
        assert np.all(
            np.around(self.prim_pred_score[id], 8) <= 1
        ), "Error in predictability computation"
        # print("Predictability Score:")
        # print(self.prim_pred_score[id])
        # print(20 * '-')
        # print('')

    def compute_pareto_front(self, id):
        leg_gr = np.less.outer(self.prim_leg_score[id], self.prim_leg_score[id])
        pred_gr = np.less.outer(self.prim_pred_score[id], self.prim_pred_score[id])
        self.pareto_front[id] = np.invert(np.any(leg_gr & pred_gr, axis=(3, 4, 5)))

    def check_if_legible(self, id):
        passing_ratio = np.max(self.current_leg_score[id]) / np.min(
            self.current_leg_score[id], where=self.current_leg_score[id] > 0, initial=1
        )
        self.is_legible[id] = passing_ratio > self.leg_tol

    def update_tau(self, id, agent):
        if (
            self.is_legible[id] and (not self.int_t[id] or self.taus[id] == 1)
        ) or agent.at_goal:
            self.taus[id] = 1
        else:
            lb = 0.0
            ub = 1.0
            tau = self.int_t[id] / max(0.01, self.pred_int_t[id])
            tau = tau * (ub - lb) + lb
            self.taus[id] = min(ub, tau)

    def update_col_mask(self, id, agent):
        intersecting = ~helper.in_front(
            self.pred_int_lines[id][0], self.int_line_heading, self.abs_prims
        )
        t_to_line = helper.directed_cost_to_line(
            self.pos, self.abs_prim_vels, self.int_lines[id], agent.vel
        )
        t_to_line = np.where(np.isposinf(t_to_line), 1e2, t_to_line)
        mask = helper.directed_intersection_pt(
            self.pos, self.abs_prim_vels, self.int_lines[id], agent.vel, t_to_line
        )
        self.col_mask |= mask[1] & intersecting

    def get_opt_prims(self):
        score = np.full((self.speed_samples, self.heading_samples), np.inf)
        for id, agent in self.interacting_agents.items():
            self.update_col_mask(id, agent)  # This might not be necessary
            new_score = (1 - self.taus[id]) * self.prim_leg_score[id] + self.taus[
                id
            ] * self.prim_pred_score[id]
            new_score = np.where(
                new_score[0] > new_score[1], new_score[0], new_score[1]
            )
            # self.pareto_front[id] = self.pareto_front[id][indices, y, x]
            score = np.minimum(score, new_score)
        # self.tot_pareto_front = np.any([self.pareto_front[id] for id in self.interacting_agents], axis=0)
        score = np.where(self.col_mask, -np.inf, score)
        # score *= self.tot_pareto_front
        # opt_prims = np.unravel_index(np.argmax(score), score.shape)
        self.opt_prims = np.argwhere(score == np.max(score))

    def get_action(self):
        self.update_abs_prims()
        self.update_abs_headings()
        self.update_abs_prim_vels()
        self.update_int_line()
        self.get_interacting_agents()
        self.remove_col_prims()
        for id, agent in self.interacting_agents.items():
            self.predict_pos(id, agent)
            self.update_pred_int_t(id, agent)
            self.get_int_costs(id, agent)
            self.compute_prim_leg(id)
            self.compute_leg(id)
            self.compute_prim_pred(id)
            # self.compute_pareto_front(id)
            self.check_if_legible(id)
            self.update_tau(id, agent)
        if self.interacting_agents:
            self.get_opt_prims()
            self.speed_idx, self.heading_idx = self.opt_prims[len(self.opt_prims) // 2]
            self.des_speed = self.speeds[self.speed_idx]
            self.des_heading = self.abs_headings[self.heading_idx]
        else:
            delta_heading = np.abs(
                self.abs_headings - helper.angle(self.goal - self.pos)
            )
            goal_cost = np.tile(delta_heading, (self.speed_samples, 1))
            goal_cost = np.where(self.col_mask, np.inf, goal_cost)
            self.speed_idx, self.heading_idx = np.unravel_index(
                np.argmin(goal_cost), goal_cost.shape
            )
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
            else:
                self.int_lines_log[id][step] = 2 * [None]
                self.pred_int_lines_log[id][step] = 2 * [None]
