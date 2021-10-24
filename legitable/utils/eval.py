import numpy as np
from prettytable import PrettyTable
from utils import helper


class Eval:
    def __init__(self, config, trial_cnt):
        self.config = config
        self.trial_cnt = trial_cnt
        self.ttg_log = {policy: np.zeros(self.trial_cnt) for policy in config.policies}
        self.extra_ttg_log = {
            policy: np.zeros(self.trial_cnt) for policy in config.policies
        }
        self.failure_log = {policy: 0 for policy in config.policies}
        self.path_efficiency_log = {
            policy: np.zeros(self.trial_cnt) for policy in config.policies
        }
        self.path_irregularity_log = {
            policy: np.zeros(self.trial_cnt) for policy in config.policies
        }
        self.legibility_log = {policy: dict() for policy in config.policies}
        self.predictability_log = {policy: dict() for policy in config.policies}

    def evaluate(self, env, iter):
        if hasattr(env.ego_agent, "time_to_goal"):
            self.ttg_log[env.ego_policy][iter] = env.ego_agent.time_to_goal
            self.extra_ttg_log[env.ego_policy][iter] = self.compute_extra_ttg(
                env.ego_agent
            )
            self.path_efficiency_log[env.ego_policy][
                iter
            ] = self.compute_path_efficiency(env.ego_agent)
        else:
            self.ttg_log[env.ego_policy][iter] = np.inf
            self.extra_ttg_log[env.ego_policy][iter] = np.inf
            self.failure_log[env.ego_policy] += 1
            self.path_efficiency_log[env.ego_policy][iter] = 0
        self.path_irregularity_log[env.ego_policy][
            iter
        ] = self.compute_path_irregularity(env.ego_agent)
        self.legibility_log[env.ego_policy][iter] = self.compute_legibility(env)
        self.predictability_log[env.ego_policy][iter] = self.compute_predictability(env)

    def compute_extra_ttg(self, agent):
        opt_ttg = (
            helper.dist(agent.start, agent.goal) - self.config.goal_tol
        ) / agent.max_speed
        return agent.time_to_goal / opt_ttg

    def compute_path_efficiency(self, agent):
        path_len = np.sum(np.linalg.norm(np.diff(agent.pos_log, axis=0), axis=-1))
        opt_path = helper.dist(agent.start, agent.goal) - agent.goal_tol
        return opt_path / path_len

    def compute_path_irregularity(self, agent):
        return np.mean(
            np.abs(agent.heading_log - helper.angle(agent.goal - agent.pos_log))
        )

    def compute_legibility(self, env):
        return legibility_score

    def compute_predictability(self, env):
        pass

    def get_summary(self):
        x = PrettyTable()
        x.field_names = [
            "Policy",
            "TTG (s)",
            "Extra TTG (%)",
            "Failure Rate (%)",
            "Path Efficiency (%)",
            "Path Irregularity (rad/m)",
            "Legibility",
        ]
        x.align["Policy"] = "l"
        for policy in self.config.policies:
            x.add_row(
                [
                    policy,
                    f"{np.mean(self.ttg_log[policy], where=self.ttg_log[policy]!=np.inf):.3f}",
                    f"{np.mean(self.extra_ttg_log[policy], where=self.ttg_log[policy]!=np.inf):.3f}",
                    f"{100 * self.failure_log[policy] / self.trial_cnt:.0f} ({self.failure_log[policy]}/{self.trial_cnt})",
                    f"{100 * np.mean(self.path_efficiency_log[policy], where=self.ttg_log[policy]!=np.inf):.0f}",
                    f"{np.mean(self.path_irregularity_log[policy]):.3f}",
                    f"{np.mean([leg_score for leg_dict in self.legibility_log[policy].values() for leg_score in leg_dict.values()]):.3f}",
                ]
            )
        if self.trial_cnt > 1:
            print(f"Average over {self.trial_cnt} trials")
        print(x)
