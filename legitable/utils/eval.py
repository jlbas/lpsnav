import numpy as np
from prettytable import PrettyTable

from utils import helper

class Eval():

    def __init__(self, config, trial_cnt):
        self.config = config
        self.trial_cnt = trial_cnt
        self.ttg_log = {policy : np.zeros(self.trial_cnt) for policy in config.policies}
        self.extra_ttg_log = {policy : np.zeros(self.trial_cnt) for policy in config.policies}
        self.failure_log = {policy : 0 for policy in config.policies}
        self.path_efficiency_log = {policy : np.zeros(self.trial_cnt) for policy in config.policies}
        self.path_irregularity_log = {policy : np.zeros(self.trial_cnt) for policy in config.policies}
        self.legibility_log = {policy : dict() for policy in config.policies}

    def evaluate(self, env, iter):
        if hasattr(env.ego_agent, 'time_to_goal'):
            self.ttg_log[env.ego_policy][iter] = env.ego_agent.time_to_goal
            self.extra_ttg_log[env.ego_policy][iter] = self.compute_extra_ttg(env.ego_agent)
            self.path_efficiency_log[env.ego_policy][iter] = self.compute_path_efficiency(env.ego_agent)
        else:
            self.ttg_log[env.ego_policy][iter] = np.inf
            self.extra_ttg_log[env.ego_policy][iter] = np.inf
            self.failure_log[env.ego_policy] += 1
            self.path_efficiency_log[env.ego_policy][iter] = 0
        self.path_irregularity_log[env.ego_policy][iter] = self.compute_path_irregularity(env.ego_agent)
        self.legibility_log[env.ego_policy][iter] = self.compute_legibility(env)
        self.legibility_log[env.ego_policy][iter] = self.compute_predictability(env)

    def compute_extra_ttg(self, agent):
        opt_ttg = (helper.dist(agent.start, agent.goal) - self.config.goal_tol) / agent.max_speed
        return agent.time_to_goal / opt_ttg

    def compute_path_efficiency(self, agent):
        path_len = np.sum(np.linalg.norm(np.diff(agent.pos_log, axis=0), axis=-1))
        opt_path = helper.dist(agent.start, agent.goal) - agent.goal_tol
        return opt_path / path_len

    def compute_path_irregularity(self, agent):
        return np.mean(np.abs(agent.heading_log - helper.angle(agent.goal - agent.pos_log)))

    def compute_legibility(self, env):
        subgoal_priors = self.config.subgoal_priors
        col_width = 2 * self.config.radius + self.config.col_buffer
        int_baseline = np.array([[0, -col_width], [0, col_width]])
        other_agents = {id : agent for id, agent in env.agents.items() if agent != env.ego_agent}
        legibility_score = {id : 0 for id in other_agents}
        rel_prims = self.config.prim_horiz * np.multiply.outer(speeds, helper.unit_vec(self.rel_headings))
        for id, agent in other_agents.items():
            goal_conditional_log = list()
            cost_tg_log = np.array()
            for i in range(len(env.ego_agent.pos_log)):
                int_line_heading = helper.wrap_to_pi(np.pi + helper.angle(env.ego_agent.goal - env.ego_agent.pos_log[i]))
                in_front = helper.in_front(agent.pos_log[i], int_line_heading, env.ego_agent.pos_log[i])
                in_radius = helper.dist(env.ego_agent.pos_log[i], agent.pos_log[i]) <= self.config.sensing_dist
                if in_front and in_radius:
                    pts = helper.rotate(int_baseline, int_line_heading)
                    int_line = agent.pos_log[i] + pts
                    cost_tg = helper.dynamic_pt_cost(env.ego_agent.pos_log[i], env.ego_agent.max_speed, int_line, int_line_heading, agent.vel_log[i])
                    if not cost_tg_log:
                        cost_tg_log = np.full((int(self.config.receding_horiz / env.timestep), 3), cost_tg)
                    cost_pg = helper.dynamic_prim_cost(env.ego_agent.pos_log[i], abs_prims, env.ego_agent.max_speed, abs_prims_vels, pred_int_line, int_line_heading, agent.vel_log[i], int_line)
                    cost_sg = cost_tg_log[-int(2 / self.config.timestep)]
                    cost_st = i * self.config.timestep
                    goal_conditional = np.exp(cost_sg - (cost_st + cost_tg)) * subgoal_priors
                    goal_conditional /= np.sum(goal_conditional)
                    goal_conditional = np.delete(goal_conditional, 1)
                    if np.any(np.isnan(goal_conditional)):
                        continue
                    goal_conditional_log.append(goal_conditional)
                    cost_tg_log.append(cost_tg)
            discounted_t = np.linspace(len(goal_conditional_log) * self.config.timestep, self.config.timestep, len(goal_conditional_log))
            legibility_score[id] = np.sum(np.max(goal_conditional_log, axis=-1) * discounted_t) / np.sum(discounted_t)
        return legibility_score

    def compute_predictability(self, env):
        pass

    def get_summary(self):
        x = PrettyTable()
        x.field_names = ["Policy", "TTG (s)", "Extra TTG (%)", "Failure Rate (%)", "Path Efficiency (%)", "Path Irregularity (rad/m)", "Legibility"]
        x.align["Policy"] = "l"
        for policy in self.config.policies:
            x.add_row([\
                policy, \
                f'{np.mean(self.ttg_log[policy], where=self.ttg_log[policy]!=np.inf):.3f}', \
                f'{np.mean(self.extra_ttg_log[policy], where=self.ttg_log[policy]!=np.inf):.3f}', \
                f'{100 * self.failure_log[policy] / self.trial_cnt:.0f} ({self.failure_log[policy]}/{self.trial_cnt})', \
                f'{100 * np.mean(self.path_efficiency_log[policy], where=self.ttg_log[policy]!=np.inf):.0f}', \
                f'{np.mean(self.path_irregularity_log[policy]):.3f}', \
                f'{np.mean([leg_score for leg_dict in self.legibility_log[policy].values() for leg_score in leg_dict.values()]):.3f}', \
            ])
        if self.trial_cnt > 1:
            print(f'Average over {self.trial_cnt} trials')
        print(x)
