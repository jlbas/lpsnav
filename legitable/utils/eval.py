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
    vals: dict[int, np.ndarray] = field(default_factory=dict)
    weights: dict[int, list] = field(default_factory=dict)


class Eval:
    def __init__(self, trial_cnt, config, scenario):
        self.config = config
        self.trial_cnt = trial_cnt
        self.scenario = scenario
        self.extra_ttg_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.failure_log = {policy: 0 for policy in config.policies}
        self.path_efficiency_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.path_irregularity_log = {
            policy: np.zeros(self.trial_cnt) for policy in config.policies
        }
        self.leg_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.pred_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.goal_inference = {
            policy: [Score() for _ in range(self.trial_cnt)] for policy in config.policies
        }
        self.traj_inference = {
            policy: [Score() for _ in range(self.trial_cnt)] for policy in config.policies
        }

    def __repr__(self):
        ret = f"{self.scenario}"
        if self.scenario == "random" or self.scenario == "circle":
            ret += f"_{self.config.num_of_agents}"
        if self.config.homogeneous:
            ret += "_homogeneous"
        if self.scenario == "random":
            ret += f"_iter_{self.config.random_scenarios}"
        return ret

    def evaluate(self, iter, env):
        if hasattr(env.ego_agent, "time_to_goal"):
            self.extra_ttg_log[env.ego_policy][iter] = self.compute_extra_ttg(env.ego_agent)
            self.path_efficiency_log[env.ego_policy][iter] = self.compute_path_efficiency(
                env.ego_agent
            )
        else:
            self.extra_ttg_log[env.ego_policy][iter] = np.inf
            self.failure_log[env.ego_policy] += 1
            self.path_efficiency_log[env.ego_policy][iter] = 0
        self.path_irregularity_log[env.ego_policy][iter] = self.compute_path_irregularity(
            env.ego_agent
        )
        (
            self.leg_log[env.ego_policy][iter],
            self.pred_log[env.ego_policy][iter],
            self.goal_inference[env.ego_policy][iter],
            self.traj_inference[env.ego_policy][iter],
        ) = self.compute_leg_pred(env)

    def compute_extra_ttg(self, agent):
        opt_ttg = (helper.dist(agent.start, agent.goal) - self.config.goal_tol) / agent.max_speed
        return (agent.time_to_goal - opt_ttg) / opt_ttg

    def compute_path_efficiency(self, agent):
        path_len = np.sum(np.linalg.norm(np.diff(agent.pos_log, axis=0), axis=-1))
        opt_path = helper.dist(agent.start, agent.goal) - agent.goal_tol
        return opt_path / path_len

    def compute_path_irregularity(self, agent):
        return np.mean(np.abs(agent.heading_log - helper.angle(agent.goal - agent.pos_log)))

    def compute_leg_pred(self, env):
        legibility = Score()
        predictability = Score()
        receding_steps = int(self.config.receding_horiz / env.timestep)
        col_width = 2 * env.ego_agent.radius + self.config.col_buffer
        int_baseline = np.array([[0, -col_width], [0, col_width]])
        cost_st = self.config.receding_horiz
        for id, agent in {id: a for id, a in env.agents.items() if a is not env.ego_agent}.items():
            legibility.vals[id] = np.zeros((len(env.ego_agent.pos_log), 2))
            predictability.vals[id] = np.zeros((len(env.ego_agent.pos_log), 2))
            start_idx = None
            for i in range(len(legibility.vals[id])):
                if not env.ego_agent.goal_log[i]:
                    int_line_heading = helper.wrap_to_pi(
                        helper.angle(env.ego_agent.pos_log[i] - env.ego_agent.goal)
                    )
                    ego_in_front = helper.in_front(
                        agent.pos_log[i], agent.heading_log[i], env.ego_agent.pos_log[i]
                    )
                    a_in_front = helper.in_front(
                        env.ego_agent.pos_log[i], env.ego_agent.heading_log[i], agent.pos_log[i]
                    )
                    in_radius = (
                        helper.dist(env.ego_agent.pos_log[i], agent.pos_log[i])
                        <= self.config.sensing_dist
                    )
                    if ego_in_front and a_in_front and in_radius:
                        if i < receding_steps:
                            receded_pos = env.ego_agent.pos_log[0] - env.ego_agent.vel_log[0] * (
                                self.config.receding_horiz - i * self.config.timestep
                            )
                        else:
                            receded_pos = env.ego_agent.pos_log[i - receding_steps]
                        scaled_speed = max(env.ego_agent.max_speed, agent.speed_log[i] + 0.1)
                        int_pts = helper.rotate(int_baseline, int_line_heading)
                        int_line = agent.pos_log[i] + int_pts
                        receded_line = int_line - agent.vel_log[i] * self.config.receding_horiz
                        cost_rg = helper.dynamic_pt_cost(
                            receded_pos,
                            scaled_speed,
                            receded_line,
                            int_line_heading,
                            agent.vel_log[i],
                        )
                        cost_tg = helper.dynamic_pt_cost(
                            env.ego_agent.pos_log[i],
                            scaled_speed,
                            int_line,
                            int_line_heading,
                            agent.vel_log[i],
                        )
                        arg = cost_rg - (cost_st + cost_tg)
                        assert np.all(np.around(arg, 1) <= 0), "Error in legibility eval"
                        goal_inference = np.exp(arg) * self.config.subgoal_priors
                        goal_inference /= np.sum(goal_inference)
                        legibility.vals[id][i] = np.delete(goal_inference, 1)
                        if start_idx is None:
                            start_idx = i
                        int_time = (i - start_idx) * env.timestep
                        start_line = int_line - agent.vel_log[i] * int_time
                        cost_sg = helper.dynamic_pt_cost(
                            env.ego_agent.pos_log[start_idx],
                            scaled_speed,
                            start_line,
                            int_line_heading,
                            agent.vel_log[i],
                        )
                        traj_arg = cost_sg - (int_time + cost_tg)
                        assert np.all(np.around(traj_arg, 0) <= 0), "Error in predictability eval"
                        with np.errstate(under="ignore", over="ignore"):
                            predictability.vals[id][i] = np.delete(np.exp(traj_arg), 1)
                        if not np.all(np.isfinite(predictability.vals[id])):
                            print("here")
                    if not a_in_front and not ego_in_front and start_idx is not None:
                        start_idx = None
            lvals = helper.split_interactions(legibility.vals[id])
            t = [np.linspace(0, len(stride) * env.timestep, len(stride)) for stride in lvals]
            for i, (sub_goal_inference, sub_t) in enumerate(zip(lvals, t)):
                discount = sub_t[-1] - sub_t if sub_t[-1] else 1
                lvals[i] = np.sum(sub_goal_inference * discount)
                lvals[i] /= np.sum(discount)
            if lvals:
                legibility.weights[id] = helper.sub_lengths(t)
                legibility.scores.append(np.average(lvals, weights=legibility.weights[id]))
                legibility.tot_weights.append(np.sum(legibility.weights[id]))
            pvals = helper.split_interactions(predictability.vals[id])
            if pvals:
                predictability.weights[id] = helper.sub_lengths(pvals)
                pvals = np.array([stride[-1] for stride in pvals])
                predictability.scores.append(np.average(pvals, weights=predictability.weights[id]))
                predictability.tot_weights.append(np.sum(predictability.weights[id]))
        if legibility.scores:
            legibility.tot_score = np.average(legibility.scores, weights=legibility.tot_weights)
        if predictability.scores:
            predictability.tot_score = np.average(
                predictability.scores, weights=predictability.tot_weights
            )
        return legibility.tot_score, predictability.tot_score, legibility.vals, predictability.vals

    def get_summary(self):
        tbl = PrettyTable()
        tbl.field_names = [
            "Policy",
            "Extra TTG (%)",
            "Failure Rate (%)",
            "Path Efficiency (%)",
            "Path Irregularity (rad/m)",
            "Legibility",
            "Predictability",
        ]
        tbl.align["Policy"] = "l"
        for policy in self.config.policies:
                extra_ttg = np.nan
                path_efficiency = np.nan
            else:
                extra_ttg = np.mean(
                    self.extra_ttg_log[policy], where=self.ttg_log[policy] != np.inf
                )
                path_efficiency = 100 * np.mean(
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
                    f"{failure_rate:.0f} ({self.failure_log[policy]}/{self.trial_cnt})",
                    f"{path_efficiency:.0f}",
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
