import os
from dataclasses import dataclass, field

import numpy as np
from prettytable import PrettyTable
from utils import helper


@dataclass
class Score:
    tot_score: float = 0
    tot_weights: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    values: dict[int, np.ndarray] = field(default_factory=dict)
    weights: dict[int, list] = field(default_factory=dict)


class Eval:
    def __init__(self, config, trial_cnt):
        self.config = config
        self.trial_cnt = trial_cnt
        self.ttg_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.extra_ttg_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.failure_log = {policy: 0 for policy in config.policies}
        self.path_efficiency_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.path_irregularity_log = {
            policy: np.zeros(self.trial_cnt) for policy in config.policies
        }
        self.leg_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.pred_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}

    def __repr__(self):
        if self.config.scenario == "random":
            return (
                f"{self.config.scenario}_{self.config.num_of_agents}_agent_"
                f"{self.config.random_scenarios}_iters"
            )
        return f"{self.config.scenario}"

    def evaluate(self, env, iter):
        if hasattr(env.ego_agent, "time_to_goal"):
            self.ttg_log[env.ego_policy][iter] = env.ego_agent.time_to_goal
            self.extra_ttg_log[env.ego_policy][iter] = self.compute_extra_ttg(env.ego_agent)
            self.path_efficiency_log[env.ego_policy][iter] = self.compute_path_efficiency(
                env.ego_agent
            )
        else:
            self.ttg_log[env.ego_policy][iter] = np.inf
            self.extra_ttg_log[env.ego_policy][iter] = np.inf
            self.failure_log[env.ego_policy] += 1
            self.path_efficiency_log[env.ego_policy][iter] = 0
        self.path_irregularity_log[env.ego_policy][iter] = self.compute_path_irregularity(
            env.ego_agent
        )
        self.leg_log[env.ego_policy][iter], self.pred_log[env.ego_policy][iter] = self.compute_lp(
            env
        )

    def compute_extra_ttg(self, agent):
        opt_ttg = (helper.dist(agent.start, agent.goal) - self.config.goal_tol) / agent.max_speed
        return agent.time_to_goal / opt_ttg

    def compute_path_efficiency(self, agent):
        path_len = np.sum(np.linalg.norm(np.diff(agent.pos_log, axis=0), axis=-1))
        opt_path = helper.dist(agent.start, agent.goal) - agent.goal_tol
        return opt_path / path_len

    def compute_path_irregularity(self, agent):
        return np.mean(np.abs(agent.heading_log - helper.angle(agent.goal - agent.pos_log)))

    def compute_lp(self, env):
        legibility = Score()
        predictability = Score()
        receding_steps = int(self.config.receding_horiz / env.timestep)
        col_width = 2 * env.ego_agent.radius + self.config.col_buffer
        int_baseline = np.array([[0, -col_width], [0, col_width]])
        cost_st = self.config.receding_horiz
        for id, agent in {id: a for id, a in env.agents.items() if a is not env.ego_agent}.items():
            legibility.values[id] = np.zeros((len(env.ego_agent.pos_log), 2))
            predictability.values[id] = np.zeros((len(env.ego_agent.pos_log), 2))
            start_idx = None
            agent.pos_log = np.pad(
                agent.pos_log, (0, len(env.ego_agent.pos_log) - len(agent.pos_log)), mode="edge"
            )
            for i in range(len(legibility.values[id])):
                int_line_heading = helper.wrap_to_pi(
                    np.pi + helper.angle(env.ego_agent.goal - env.ego_agent.pos_log[i])
                )
                in_front = helper.in_front(
                    agent.pos_log[i], int_line_heading, env.ego_agent.pos_log[i]
                )
                in_radius = (
                    helper.dist(env.ego_agent.pos_log[i], agent.pos_log[i])
                    <= self.config.sensing_dist
                )
                if in_front and in_radius:
                    int_pts = helper.rotate(int_baseline, int_line_heading)
                    int_line = agent.pos_log[i] + int_pts
                    if i < receding_steps:
                        ego_pos = env.ego_agent.pos_log[i]
                        ego_vel = env.ego_agent.vel_log[i]
                        receded_pos = ego_pos - ego_vel * self.config.receding_horiz
                    else:
                        receded_pos = env.ego_agent.pos_log[i - receding_steps]
                    receded_line = int_line - agent.vel_log[i] * self.config.receding_horiz
                    cost_rg = helper.dynamic_pt_cost(
                        receded_pos,
                        self.config.scaled_speed,
                        receded_line,
                        int_line_heading,
                        agent.vel_log[i],
                    )
                    cost_tg = helper.dynamic_pt_cost(
                        env.ego_agent.pos_log[i],
                        self.config.scaled_speed,
                        int_line,
                        int_line_heading,
                        agent.vel_log[i],
                    )
                    arg = cost_rg - (cost_st + cost_tg)
                    goal_inference = np.exp(arg) * self.config.subgoal_priors
                    goal_inference /= np.sum(goal_inference)
                    legibility.values[id][i] = np.delete(goal_inference, 1)
                    if start_idx is None:
                        start_idx = i
                    int_time = (i - start_idx) * env.timestep
                    start_line = int_line - agent.vel_log[i] * int_time
                    cost_sg = helper.dynamic_pt_cost(
                        env.ego_agent.pos_log[start_idx],
                        self.config.max_speed,
                        start_line,
                        int_line_heading,
                        agent.vel_log[i],
                    )
                    true_cost_tg = helper.dynamic_pt_cost(
                        env.ego_agent.pos_log[i],
                        self.config.max_speed,
                        int_line,
                        int_line_heading,
                        agent.vel_log[i],
                    )
                    traj_arg = cost_sg - (int_time + true_cost_tg)
                    with np.errstate(under="ignore", over="ignore"):
                        predictability.values[id][i] = np.delete(np.exp(traj_arg), 1)
                if not in_front and start_idx is not None:
                    start_idx = None
            legibility.values[id] = np.max(legibility.values[id], axis=-1)
            zero_mask = legibility.values[id] == 0
            split_idx = (np.argwhere(np.diff(zero_mask.astype(int))) + 1).flatten()
            legibility.values[id] = np.split(legibility.values[id], split_idx)
            legibility.values[id] = helper.remove_zero_arrs(legibility.values[id])
            t = [
                np.linspace(0, len(stride) * env.timestep, len(stride))
                for stride in legibility.values[id]
            ]
            for i, (sub_goal_inference, sub_t) in enumerate(zip(legibility.values[id], t)):
                discount = sub_t[-1] - sub_t if sub_t[-1] else 1
                legibility.values[id][i] = np.sum(sub_goal_inference * discount)
                legibility.values[id][i] /= np.sum(discount)
            if legibility.values[id]:
                legibility.weights[id] = [len(stride) for stride in t]
                legibility.scores.append(
                    np.average(legibility.values[id], weights=legibility.weights[id])
                )
                legibility.tot_weights.append(np.sum(legibility.weights[id]))
            predictability.values[id] = np.max(predictability.values[id], axis=-1)
            zero_mask = predictability.values[id] == 0
            split_idx = (np.argwhere(np.diff(zero_mask.astype(int))) + 1).flatten()
            predictability.values[id] = np.split(predictability.values[id], split_idx)
            predictability.values[id] = helper.remove_zero_arrs(predictability.values[id])
            if predictability.values[id]:
                predictability.weights[id] = [len(stride) for stride in predictability.values[id]]
                predictability.values[id] = np.array(
                    [stride[-1] for stride in predictability.values[id]]
                )
                assert np.all(
                    np.around(predictability.values[id], 8) <= 1
                ), "Error in trajectory inference computation"
                predictability.scores.append(
                    np.average(predictability.values[id], weights=predictability.weights[id])
                )
                predictability.tot_weights.append(np.sum(predictability.weights[id]))
        legibility.tot_score = np.average(legibility.scores, weights=legibility.tot_weights)
        predictability.tot_score = np.average(
            predictability.scores, weights=predictability.tot_weights
        )
        return legibility.tot_score, predictability.tot_score

    def get_summary(self):
        tbl = PrettyTable()
        tbl.field_names = [
            "Policy",
            "TTG (s)",
            "Extra TTG (%)",
            "Failure Rate (%)",
            "Path Efficiency (%)",
            "Path Irregularity (rad/m)",
            "Legibility",
            "Predictability",
        ]
        tbl.align["Policy"] = "l"
        for policy in self.config.policies:
            if np.all(self.ttg_log[policy] == np.inf):
                ttg = np.nan
                extra_ttg = np.nan
                path_efficiency = np.nan
            else:
                ttg = np.mean(self.ttg_log[policy], where=self.ttg_log[policy] != np.inf)
                extra_ttg = np.mean(
                    self.extra_ttg_log[policy], where=self.ttg_log[policy] != np.inf
                )
                path_efficiency = 100 * np.mean(
                    self.path_efficiency_log[policy], where=self.ttg_log[policy] != np.inf
                )
            failure_rate = 100 * self.failure_log[policy] / self.trial_cnt
            path_irregularity = np.mean(self.path_irregularity_log[policy])
            legibility = np.mean(self.leg_log[policy])
            predictability = np.mean(self.pred_log[policy])
            tbl.add_row(
                [
                    policy,
                    f"{ttg:.3f}",
                    f"{extra_ttg:.3f}",
                    f"{path_efficiency:.0f}",
                    f"{failure_rate:.0f} ({self.failure_log[policy]}/{self.trial_cnt})",
                    f"{path_irregularity:.3f}",
                    f"{legibility:.3f}",
                    f"{predictability:.3f}",
                ]
            )
        if self.trial_cnt > 1:
            print(f"Average over {self.trial_cnt} trials")
        print(tbl)

        if self.config.save_tbl:
            os.makedirs(self.config.tbl_dir, exist_ok=True)
            with open(os.path.join(self.config.tbl_dir, f"{str(self)}.latex"), "w") as f:
                f.write(tbl.get_latex_string())
