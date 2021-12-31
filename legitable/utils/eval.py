import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter
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
        self.conf = config
        self.trial_cnt = trial_cnt
        self.scenario = scenario
        self.colors = {policy: "" for policy in self.conf.env.policies}
        self.extra_ttg_log = {policy: np.zeros(self.trial_cnt) for policy in self.conf.env.policies}
        self.failure_log = {policy: 0 for policy in self.conf.env.policies}
        self.path_efficiency_log = {
            policy: np.zeros(self.trial_cnt) for policy in self.conf.env.policies
        }
        self.path_irregularity_log = {
            policy: np.zeros(self.trial_cnt) for policy in self.conf.env.policies
        }
        if self.scenario != "random" and self.scenario != "circle":
            self.leg_log = {policy: np.zeros(self.trial_cnt) for policy in self.conf.env.policies}
            self.pred_log = {policy: np.zeros(self.trial_cnt) for policy in self.conf.env.policies}
            self.goal_inference = {
                policy: [dict() for _ in range(self.trial_cnt)] for policy in self.conf.env.policies
            }
            self.traj_inference = {
                policy: [dict() for _ in range(self.trial_cnt)] for policy in self.conf.env.policies
            }
        self.nav_contrib_log = {
            policy: np.zeros(self.trial_cnt) for policy in self.conf.env.policies
        }
        self.init_symbols()

    def __repr__(self):
        ret = f"{self.scenario}"
        if self.scenario in ("random", "circle"):
            ret += f"_{self.conf.env.num_of_agents}"
        if self.conf.env.homogeneous:
            ret += "_homogeneous"
        if self.scenario == "random":
            ret += f"_iter_{self.conf.env.random_scenarios}"
        return ret

    def init_symbols(self):
        prx, pry, phx, phy, vr, thr, vh, thh = smp.symbols("prx pry phx phy vr thr vh thh")
        dvx = vh * smp.cos(thh) - vr * smp.cos(thr)
        dvy = vr * smp.sin(thr) - vh * smp.sin(thh)
        dpx = phx - prx
        dpy = phy - pry
        args = (prx, pry, phx, phy, vr, thr, vh, thh)
        mpd = smp.sqrt((dvx * dpy + dvy * dpx) ** 2 / (dvx ** 2 + dvy ** 2))
        self.mpd = smp.lambdify(args, mpd)
        self.mpd_partials = [smp.lambdify(args, smp.diff(mpd, p)) for p in [vr, thr, vh, thh]]

    def evaluate(self, iter, env):
        self.colors[env.ego_policy] = env.ego_agent.color
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
        self.get_interactions(env)
        if hasattr(self, "leg_log"):
            (
                self.leg_log[env.ego_policy][iter],
                self.pred_log[env.ego_policy][iter],
                self.goal_inference[env.ego_policy][iter],
                self.traj_inference[env.ego_policy][iter],
            ) = self.compute_leg_pred(env)
        self.compute_mpd(env)
        self.nav_contrib_log[env.ego_policy][iter] = self.compute_nav_contrib(env)
        if self.conf.eval.show_nav_contrib_plot or self.conf.eval.save_nav_contrib_plot:
            self.make_nav_contrib_plot(env)

    def compute_extra_ttg(self, agent):
        opt_ttg = (
            helper.dist(agent.start, agent.goal) - self.conf.agent.goal_tol
        ) / agent.max_speed
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
        col_width = 2 * env.ego_agent.radius + self.conf.lpnav.col_buffer
        receding_steps = int(self.conf.lpnav.receding_horiz / env.dt)
        int_baseline = np.array([[0, -col_width], [0, col_width]])
        cost_st = self.conf.lpnav.receding_horiz
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
                        <= self.conf.agent.sensing_dist
                    )
                    if ego_in_front and a_in_front and in_radius:
                        if i < receding_steps:
                            receded_pos = env.ego_agent.pos_log[0] - env.ego_agent.vel_log[0] * (
                                self.conf.lpnav.receding_horiz - i * self.conf.env.dt
                            )
                        else:
                            receded_pos = env.ego_agent.pos_log[i - receding_steps]
                        scaled_speed = max(env.ego_agent.max_speed, agent.speed_log[i] + 0.1)
                        int_pts = helper.rotate(int_baseline, int_line_heading)
                        int_line = agent.pos_log[i] + int_pts
                        receded_line = int_line - agent.vel_log[i] * self.conf.lpnav.receding_horiz
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
                        goal_inference = np.exp(arg) * self.conf.lpnav.subgoal_priors
                        goal_inference /= np.sum(goal_inference)
                        legibility.vals[id][i] = np.delete(goal_inference, 1)
                        if start_idx is None:
                            start_idx = i
                        int_time = (i - start_idx) * env.dt
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
            t = [np.linspace(0, len(stride) * env.dt, len(stride)) for stride in lvals]
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

    def get_interactions(self, env):
        self.int_idx = dict()
        for id, agent in env.ego_agent.other_agents.items():
            inter_dist = helper.dist(env.ego_agent.pos_log, agent.pos_log)
            if np.any(env.ego_agent.col_log):
                end_idx = np.minimum(np.argmax(env.ego_agent.col_log), np.argmin(inter_dist))
            elif np.any(env.ego_agent.goal_log):
                end_idx = np.minimum(np.argmax(env.ego_agent.goal_log), np.argmin(inter_dist))
            else:
                end_idx = np.argmin(inter_dist)
            start_idx = np.argmax(
                helper.in_front(env.ego_agent.pos_log, env.ego_agent.heading_log, agent.pos_log)
            )
            self.int_idx[id] = [start_idx, end_idx]

    def compute_mpd(self, env):
        self.mpd_params = dict()
        self.mpd_args = dict()
        self.mpd_vals = dict()
        for id, agent in env.ego_agent.other_agents.items():
            self.mpd_params[id] = [
                env.ego_agent.speed_log[self.int_idx[id][0] : self.int_idx[id][1]],
                env.ego_agent.heading_log[self.int_idx[id][0] : self.int_idx[id][1]],
                agent.speed_log[self.int_idx[id][0] : self.int_idx[id][1]],
                agent.heading_log[self.int_idx[id][0] : self.int_idx[id][1]],
            ]
            self.mpd_args[id] = [
                env.ego_agent.pos_log[:, 0][self.int_idx[id][0] : self.int_idx[id][1]],
                env.ego_agent.pos_log[:, 1][self.int_idx[id][0] : self.int_idx[id][1]],
                agent.pos_log[:, 0][self.int_idx[id][0] : self.int_idx[id][1]],
                agent.pos_log[:, 1][self.int_idx[id][0] : self.int_idx[id][1]],
                *self.mpd_params[id],
            ]
            with np.errstate(invalid="ignore"):
                self.mpd_vals[id] = gaussian_filter(self.mpd(*self.mpd_args[id]), sigma=3)

    def compute_nav_contrib(self, env):
        self.eps = dict()
        nav_contrib = list()
        for id in env.ego_agent.other_agents:
            if np.all([a.size != 0 for a in self.mpd_params[id]]):
                dparams = [np.diff(np.pad(p, (1, 0), mode="edge")) for p in self.mpd_params[id]]
                dparams[1::2] = helper.wrap_to_pi(dparams[1::2])
                with np.errstate(invalid="ignore"):
                    self.eps[id] = [
                        gaussian_filter(partial(*self.mpd_args[id]) * dp, sigma=3)
                        for partial, dp in zip(self.mpd_partials, dparams)
                    ]
                cum_contrib = [
                    np.trapz(e / self.conf.env.dt, dx=self.conf.env.dt) for e in self.eps[id]
                ]
                if np.any(cum_contrib):
                    nav_contrib.append(sum(cum_contrib[:2]) / sum(cum_contrib))
        return -np.inf if not nav_contrib else np.mean(nav_contrib)

    def make_nav_contrib_plot(self, env):
        self.conf.animation.dark_bg and plt.style.use("dark_background")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(16, 4)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("Time (s)")
        for id, agent in env.ego_agent.other_agents.items():
            t = np.linspace(0, self.conf.env.dt * len(self.mpd_vals[id]), len(self.mpd_vals[id]))
            ax1.plot(t, self.mpd_vals[id], label=r"$MPD(t)$", c=agent.color, lw=2)
            ax2.plot(
                t,
                self.eps[id][0],
                c=env.ego_agent.color,
                lw=2,
                label=r"$\frac{\partial MPD}{\partial v_R}$",
            )
            ax2.plot(
                t,
                self.eps[id][1],
                c=env.ego_agent.color,
                lw=2,
                ls="--",
                label=r"$\frac{\partial MPD}{\partial \theta_R}$",
            )
            ax2.plot(
                t,
                self.eps[id][2],
                c=agent.color,
                lw=2,
                label=r"$\frac{\partial MPD}{\partial v_H}$",
            )
            ax2.plot(
                t,
                self.eps[id][3],
                c=agent.color,
                lw=2,
                ls="--",
                label=r"$\frac{\partial MPD}{\partial \theta_H}$",
            )
            ax3.plot(t, env.dt * np.cumsum(self.eps[id][0]), c=env.ego_agent.color)
            ax3.plot(t, env.dt * np.cumsum(self.eps[id][1]), c=env.ego_agent.color, ls="--", lw=2)
            ax3.plot(t, env.dt * np.cumsum(self.eps[id][2]), c="gray", lw=2)
            ax3.plot(t, env.dt * np.cumsum(self.eps[id][3]), c="gray", ls="--", lw=2)
        fig.suptitle(str(env).replace("_", " "))
        ax1.set_ylabel(r"$MPD(t)$ (m)")
        ax1.legend()
        ax2.set_ylabel("Instantaneous effect on MPD (m/s)")
        ax2.legend()
        ax3.set_ylabel("Cumulative effect on MPD (m)")
        if self.conf.eval.save_nav_contrib_plot:
            dir = self.conf.eval.nav_contrib_plot_dir
            os.makedirs(dir, exist_ok=True)
            fig.savefig(os.path.join(dir, f"{str(env)}.pdf"), backend="pgf")
        if self.conf.eval.show_nav_contrib_plot:
            plt.show()

    def get_table(self):
        clean_policies = {
            "lpnav": "LPNav",
            "social_momentum": "SM",
            "sa_cadrl": "SA-CADRL",
            "ga3c_cadrl": "GA3C-CADRL",
            "rvo": "ORCA",
        }
        tbl_dict = {
            "policy": {
                "header": "Policy",
                "vals": [clean_policies[p] for p in self.conf.env.policies],
            },
            "extra_ttg": {
                "header": "Extra TTG ($\\%$)",
                "decimals": 2,
                "function": min,
                "raw_vals": [],
                "stds": [],
                "vals": [],
            },
            "failure_rate": {
                "header": "Failure Rate ($\\%$)",
                "decimals": 0,
                "function": min,
                "raw_vals": [],
                "vals": [],
            },
            "path_efficiency": {
                "header": "Path Efficiency ($\\%$)",
                "decimals": 2,
                "function": max,
                "raw_vals": [],
                "stds": [],
                "vals": [],
            },
            "path_irregularity": {
                "header": "Path Irregularity (rad/m)",
                "decimals": 4,
                "function": min,
                "raw_vals": [],
                "stds": [],
                "vals": [],
            },
            "nav_contrib": {
                "header": "Navigation Contribution ($\\%$)",
                "decimals": 2,
                "function": max,
                "raw_vals": [],
                "stds": [],
                "vals": [],
            },
        }
        if hasattr(self, "leg_log"):
            tbl_dict.update(
                {
                    "legibility": {
                        "header": "Legibility ($\\%$)",
                        "decimals": 2,
                        "function": max,
                        "raw_vals": [],
                        "stds": [],
                        "vals": [],
                    },
                    "predictability": {
                        "header": "Predictability ($\\%$)",
                        "decimals": 2,
                        "function": max,
                        "raw_vals": [],
                        "stds": [],
                        "vals": [],
                    },
                }
            )

        for policy in self.conf.env.policies:
            if np.all(self.extra_ttg_log[policy] == np.inf):
                tbl_dict["extra_ttg"]["raw_vals"].append(np.nan)
                tbl_dict["extra_ttg"]["stds"].append(np.nan)
                tbl_dict["path_efficiency"]["raw_vals"].append(np.nan)
                tbl_dict["path_efficiency"]["stds"].append(np.nan)
            else:
                tbl_dict["extra_ttg"]["raw_vals"].append(
                    100
                    * np.mean(
                        self.extra_ttg_log[policy], where=self.extra_ttg_log[policy] != np.inf
                    )
                )
                tbl_dict["extra_ttg"]["stds"].append(
                    np.std(self.extra_ttg_log[policy], where=self.extra_ttg_log[policy] != np.inf)
                )
                tbl_dict["path_efficiency"]["raw_vals"].append(
                    100
                    * np.mean(
                        self.path_efficiency_log[policy],
                        where=self.path_efficiency_log[policy] != 0,
                    )
                )
                tbl_dict["path_efficiency"]["stds"].append(
                    np.std(
                        self.path_efficiency_log[policy],
                        where=self.path_efficiency_log[policy] != 0,
                    )
                )
            tbl_dict["failure_rate"]["raw_vals"].append(
                100 * self.failure_log[policy] / self.trial_cnt
            )
            tbl_dict["path_irregularity"]["raw_vals"].append(
                np.mean(self.path_irregularity_log[policy])
            )
            tbl_dict["path_irregularity"]["stds"].append(np.std(self.path_irregularity_log[policy]))
            if hasattr(self, "leg_log"):
                tbl_dict["legibility"]["raw_vals"].append(100 * np.nanmean(self.leg_log[policy]))
                tbl_dict["legibility"]["stds"].append(100 * np.nanstd(self.leg_log[policy]))
                tbl_dict["predictability"]["raw_vals"].append(
                    100 * np.nanmean(self.pred_log[policy])
                )
                tbl_dict["predictability"]["stds"].append(100 * np.nanstd(self.pred_log[policy]))
            tbl_dict["nav_contrib"]["raw_vals"].append(
                100 * np.nanmean(self.nav_contrib_log[policy])
            )
            tbl_dict["nav_contrib"]["stds"].append(100 * np.nanstd(self.nav_contrib_log[policy]))

        for k, v in tbl_dict.items():
            if "function" in v:
                tbl_dict[k]["opt"] = tbl_dict[k]["function"](tbl_dict[k]["raw_vals"])
                for val in tbl_dict[k]["raw_vals"]:
                    formatted_val = f"{val:.{tbl_dict[k]['decimals']}f}"
                    if val == tbl_dict[k]["opt"]:
                        formatted_val = f"\\textbf{{{formatted_val}}}"
                    tbl_dict[k]["vals"].append(formatted_val)

        return tbl_dict

    def show_tbl(self, tbl_dict):
        if self.trial_cnt > 1:
            print(f"Average over {self.trial_cnt} trials")
        tbl = PrettyTable()
        for col in tbl_dict.values():
            tbl.add_column(col["header"], col["vals"])
        print(tbl)

    def save_tbl(self, tbl_dict):
        rows = []
        for i in range(len(self.conf.env.policies)):
            row = []
            for v in tbl_dict.values():
                row.append(v["vals"][i])
            rows.append(row)
        os.makedirs(self.conf.eval.tbl_dir, exist_ok=True)
        with open(os.path.join(self.conf.eval.tbl_dir, f"{str(self)}.tex"), "w") as f:
            newline_char = " \\\\\n\t\t\t\t"
            f.write(
                f"""
                \\begin{{table*}}
                \\caption{{Performance metrics averaged over {self.conf.env.random_scenarios} random scenarios with {self.conf.env.num_of_agents} {"homogeneous " if self.conf.env.homogeneous else ""}agents}}
                \\label{{tbl:results}}
                \\begin{{tabularx}}{{\\textwidth}}{{@{{}}X*{{{(len(tbl_dict) - 1)}}}{{Y}}@{{}}}}
                    \\toprule
                    {" & ".join([v["header"] for v in tbl_dict.values()])} \\\\
                    \\midrule
                    {newline_char.join([" & ".join(row) for row in rows])} \\\\
                    \\bottomrule
                \\end{{tabularx}}
                \\end{{table*}}
                """.strip()
            )

    def make_bar_chart(self, tbl_dict):
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times"],
            }
        )
        for k, metric in tbl_dict.items():
            if "stds" in metric:
                fig, ax = plt.subplots(constrained_layout=True)
                fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
                fig.set_size_inches(3, 2)
                policies = []
                color = []
                height = []
                yerr = []
                for policy, c, val, std in zip(
                    tbl_dict["policy"]["vals"],
                    self.colors.values(),
                    metric["raw_vals"],
                    metric["stds"],
                ):
                    if np.isfinite(val):
                        policies.append(policy)
                        color.append(c)
                        height.append(val)
                        yerr.append(std)
                ax.bar(
                    np.arange(len(policies)),
                    height,
                    yerr=yerr,
                    color=color,
                    tick_label=policies,
                    capsize=10,
                )
                ax.set_ylabel(metric["header"])
                self.conf.eval.show_bar_chart and plt.show()
                if self.conf.eval.save_bar_chart:
                    os.makedirs(self.conf.eval.bar_chart_dir, exist_ok=True)
                    filename = f"{str(self)}.pdf"
                    fullpath = os.path.join(self.conf.eval.bar_chart_dir, filename)
                    plt.savefig(fullpath, backend="pgf")

    def get_summary(self):
        tbl_dict = self.get_table()
        self.conf.eval.show_tbl and self.show_tbl(tbl_dict)
        self.conf.eval.save_tbl and self.save_tbl(tbl_dict)
        self.make_bar_chart(tbl_dict)
