import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter
from utils import helper

from legitable.utils.opt_traj import get_opt_traj


@dataclass
class Score:
    tot_score: float = np.nan
    vals: dict[int, np.ndarray] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)


class Eval:
    def __init__(self, trial_cnt, config, scenario):
        self.conf = config
        self.trial_cnt = trial_cnt
        self.scenario = scenario
        self.colors = {p: "" for p in self.conf.env.policies}
        self.extra_ttg_log = {p: np.full(self.trial_cnt, np.inf) for p in self.conf.env.policies}
        self.failure_mask = {p: np.full(self.trial_cnt, False) for p in self.conf.env.policies}
        self.path_efficiency_log = {
            p: np.full(self.trial_cnt, np.inf) for p in self.conf.env.policies
        }
        self.path_irregularity_log = {
            p: np.full(self.trial_cnt, np.inf) for p in self.conf.env.policies
        }
        if self.scenario != "random" and self.scenario != "circle":
            self.leg_log = {p: np.full(self.trial_cnt, np.inf) for p in self.conf.env.policies}
            self.pred_log = {p: np.full(self.trial_cnt, np.inf) for p in self.conf.env.policies}
            self.goal_inference = {
                p: [dict() for _ in range(self.trial_cnt)] for p in self.conf.env.policies
            }
            self.traj_inference = {
                p: [dict() for _ in range(self.trial_cnt)] for p in self.conf.env.policies
            }
        self.nav_contrib_log = {p: np.full(self.trial_cnt, np.inf) for p in self.conf.env.policies}
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
        return 0 if not path_len else opt_path / path_len

    def compute_path_irregularity(self, agent):
        return np.mean(np.abs(agent.heading_log - helper.angle(agent.goal - agent.pos_log)))

    def get_interactions(self, env):
        self.int_idx = dict()
        self.int_slice = dict()
        self.int_t = dict()
        self.int_line = dict()
        self.int_line_heading = helper.wrap_to_pi(
            helper.angle(env.ego_agent.pos_log - env.ego_agent.goal)
        )
        for id, agent in env.ego_agent.other_agents.items():
            in_front = helper.in_front(agent.pos_log, self.int_line_heading, env.ego_agent.pos_log)
            inter_dist = helper.dist(env.ego_agent.pos_log, agent.pos_log)
            in_radius = inter_dist < self.conf.agent.sensing_dist
            col_width = env.ego_agent.radius + agent.radius
            rel_int_line = np.array([[0, -col_width], [0, col_width]])
            abs_int_line = helper.rotate(rel_int_line, self.int_line_heading)
            self.int_line[id] = abs_int_line + agent.pos_log
            feasible = (
                helper.cost_to_line(
                    env.ego_agent.pos_log, env.ego_agent.max_speed, self.int_line[id], agent.vel_log
                )
                < 1e2
            )
            is_interacting = in_front & in_radius & feasible
            if ~np.any(is_interacting) or env.dt * np.sum(is_interacting) < 1:
                start_idx, end_idx = (0, 0)
            else:
                start_idx = np.argmax(is_interacting)
                end_idx = np.nonzero(is_interacting)[0][-1]
            self.int_idx[id] = [start_idx, end_idx]
            self.int_slice[id] = slice(start_idx, end_idx + 1)
            self.int_t[id] = env.dt * np.arange(start_idx, end_idx + 1)

    def compute_int_costs(self, dt, ego_agent, id, agent, receding_steps):
        init_receded_pos = (
            ego_agent.pos_log[0]
            - dt * np.arange(receding_steps + 1, 1, -1)[:, None] * ego_agent.vel_log[0]
        )
        receded_pos = np.concatenate((init_receded_pos, ego_agent.pos_log[:-receding_steps]))[
            self.int_slice[id]
        ]
        receded_line = (
            self.int_line[id][:, self.int_slice[id]]
            - agent.vel_log[self.int_slice[id]] * self.conf.lpnav.receding_horiz
        )
        receded_start_line = (
            self.int_line[id][:, self.int_slice[id]]
            - agent.vel_log[self.int_slice[id]] * self.int_t[id][:, None]
        )
        scaled_speed = max(ego_agent.max_speed, agent.max_speed + 0.1)
        cost_rg = helper.dynamic_pt_cost(
            receded_pos,
            scaled_speed,
            receded_line,
            self.int_line_heading[self.int_slice[id]],
            agent.vel_log[self.int_slice[id]],
        )
        cost_tg = helper.dynamic_pt_cost(
            ego_agent.pos_log[self.int_slice[id]],
            scaled_speed,
            self.int_line[id][:, self.int_slice[id]],
            self.int_line_heading[self.int_slice[id]],
            agent.vel_log[self.int_slice[id]],
        )
        cost_rtg = self.conf.lpnav.receding_horiz + cost_tg
        cost_sg = helper.dynamic_pt_cost(
            ego_agent.pos_log[self.int_idx[id][0]],
            scaled_speed,
            receded_start_line,
            self.int_line_heading[self.int_slice[id]],
            agent.vel_log[self.int_slice[id]],
        )
        return (cost_rg, cost_rtg, cost_tg, cost_sg)

    def compute_leg_pred(self, env):
        legibility = Score()
        predictability = Score()
        receding_steps = int(self.conf.lpnav.receding_horiz / env.dt)
        for id, agent in env.ego_agent.other_agents.items():
            if np.diff(self.int_idx[id]) > 1:
                cost_rg, cost_rtg, cost_tg, cost_sg = self.compute_int_costs(
                    env.dt, env.ego_agent, id, agent, receding_steps
                )
                arg = cost_rg - cost_rtg
                # arg = np.where(cost_rtg > 100, -np.inf, arg)
                # arg = np.clip(arg, -100, 0)
                with np.errstate(under="ignore"):
                    legibility.vals[id] = np.exp(arg) * np.expand_dims(
                        self.conf.lpnav.subgoal_priors, axis=-1
                    )
                legibility.vals[id] /= np.sum(legibility.vals[id], axis=0)
                legibility.vals[id] = np.delete(legibility.vals[id], 1, 0)
                num = np.trapz(
                    self.int_t[id][::-1] * np.max(legibility.vals[id], axis=0), dx=env.dt
                )
                den = np.trapz(self.int_t[id][::-1], dx=env.dt)
                legibility.scores[id] = num / den
                cost_stg = self.int_t[id] + cost_tg
                arg = cost_sg - cost_stg
                # arg = np.where(cost_stg > 100, -np.inf, arg)
                # arg = np.clip(arg, -100, 0)
                with np.errstate(under="ignore"):
                    predictability.vals[id] = np.delete(np.exp(arg), 1, 0)
                predictability.scores[id] = np.max(predictability.vals[id][:, -1])
                if self.conf.eval.normalize_lp:
                    min_leg, max_leg = get_opt_traj(
                        self.int_idx[id],
                        self.int_slice[id],
                        env.dt,
                        env.ego_agent,
                        agent,
                        receding_steps,
                        self.conf.lpnav.receding_horiz,
                        self.conf.lpnav.subgoal_priors,
                    )
                    if min_leg is not None and max_leg is not None:
                        legibility.scores[id] = (legibility.scores[id] - min_leg) / (
                            max_leg - min_leg
                        )
                        if np.any(legibility.scores[id] > 1):
                            print("Error in L comp")
                    else:
                        legibility.scores[id] = np.nan
        if legibility.scores:
            legibility.tot_score = np.mean([s for s in legibility.scores.values()])
            predictability.tot_score = np.mean([s for s in predictability.scores.values()])
        return legibility.tot_score, predictability.tot_score, legibility.vals, predictability.vals

    def compute_mpd(self, env):
        self.mpd_params = dict()
        self.mpd_args = dict()
        self.mpd_vals = dict()
        for id, agent in env.ego_agent.other_agents.items():
            self.mpd_params[id] = [
                env.ego_agent.speed_log[self.int_slice[id]],
                env.ego_agent.heading_log[self.int_slice[id]],
                agent.speed_log[self.int_slice[id]],
                agent.heading_log[self.int_slice[id]],
            ]
            self.mpd_args[id] = [
                env.ego_agent.pos_log[:, 0][self.int_slice[id]],
                env.ego_agent.pos_log[:, 1][self.int_slice[id]],
                agent.pos_log[:, 0][self.int_slice[id]],
                agent.pos_log[:, 1][self.int_slice[id]],
                *self.mpd_params[id],
            ]
            with np.errstate(invalid="ignore"):
                self.mpd_vals[id] = gaussian_filter(self.mpd(*self.mpd_args[id]), sigma=3)

    def compute_nav_contrib(self, env):
        self.eps = dict()
        nav_contrib = [np.nan]
        for id in env.ego_agent.other_agents:
            if np.all([a.size != 0 for a in self.mpd_params[id]]):
                dparams = [np.diff(np.pad(p, (1, 0), mode="edge")) for p in self.mpd_params[id]]
                dparams[1::2] = helper.wrap_to_pi(dparams[1::2])
                with np.errstate(invalid="ignore"):
                    self.eps[id] = [
                        gaussian_filter(partial(*self.mpd_args[id]) * dp, sigma=3)
                        for partial, dp in zip(self.mpd_partials, dparams)
                    ]
                if np.all(np.isfinite(self.eps[id])):
                    cum_contrib = [
                        np.trapz(e / self.conf.env.dt, dx=self.conf.env.dt) for e in self.eps[id]
                    ]
                    if np.any(cum_contrib):
                        nav_contrib.append(sum(cum_contrib[:2]) / sum(cum_contrib))
        return np.nanmean(nav_contrib)

    def make_nav_contrib_plot(self, env):
        self.conf.animation.dark_bg and plt.style.use("dark_background")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(16, 4)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("Time (s)")
        for id, agent in env.ego_agent.other_agents.items():
            if self.mpd_vals[id].size != 0:
                t = np.linspace(
                    0, self.conf.env.dt * len(self.mpd_vals[id]), len(self.mpd_vals[id])
                )
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
                ax3.plot(
                    t, env.dt * np.cumsum(self.eps[id][1]), c=env.ego_agent.color, ls="--", lw=2
                )
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
            "sfm": "SFM",
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
        newline_char = " \\\\\n\t\t\t\t"
        contents = f"""\\begin{{table*}}
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
"""
        with open(os.path.join(self.conf.eval.tbl_dir, f"{str(self)}.tex"), "w") as f:
            f.write(contents)

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
