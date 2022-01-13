from copy import deepcopy
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter
from utils import helper

from legitable.utils.opt_traj import get_opt_traj


def eval_extra_ttg(ego_agent, goal_tol):
    if hasattr(ego_agent, "time_to_goal"):
        opt_ttg = (helper.dist(ego_agent.start, ego_agent.goal) - goal_tol) / ego_agent.max_speed
        return (ego_agent.time_to_goal - opt_ttg) / opt_ttg
    return np.inf


def eval_failure(ego_agent):
    return False if hasattr(ego_agent, "time_to_goal") else True


def eval_efficiency(ego_agent):
    if hasattr(ego_agent, "time_to_goal"):
        path_len = np.sum(np.linalg.norm(np.diff(ego_agent.pos_log, axis=0), axis=-1))
        opt_path = helper.dist(ego_agent.start, ego_agent.goal) - ego_agent.goal_tol
        return 0 if not path_len else opt_path / path_len
    return 0


def eval_irregularity(ego_agent):
    return np.mean(np.abs(ego_agent.heading_log - helper.angle(ego_agent.goal - ego_agent.pos_log)))


def eval_legibility(dt, ego_agent, interaction, int_cost, config):
    legibility = Score()
    for id, agent in ego_agent.other_agents.items():
        if np.diff(interaction.int_idx[id]) > 1:
            arg = int_cost.rg[id] - int_cost.rtg[id]
            arg = np.where(int_cost.rtg[id] > config.lpnav.max_cost, -np.inf, arg)
            arg = np.clip(arg, -config.lpnav.max_cost, 0)
            legibility.vals[id] = np.exp(arg) * np.expand_dims(config.lpnav.subgoal_priors, axis=-1)
            legibility.vals[id] /= np.sum(legibility.vals[id], axis=0)
            legibility.vals[id] = np.delete(legibility.vals[id], 1, 0)
            num = np.trapz(interaction.int_t[id][::-1] * np.max(legibility.vals[id], axis=0), dx=dt)
            den = np.trapz(interaction.int_t[id][::-1], dx=dt)
            legibility.scores[id] = num / den
            if config.eval.normalize_lp:
                min_leg, max_leg = get_opt_traj(
                    interaction.int_idx[id],
                    interaction.int_slice[id],
                    dt,
                    ego_agent,
                    agent,
                    config.lpnav.receding_horiz,
                    config.lpnav.subgoal_priors,
                )
                if min_leg is not None and max_leg is not None:
                    legibility.scores[id] = (legibility.scores[id] - min_leg) / (max_leg - min_leg)
                else:
                    legibility.scores[id] = np.nan
    if legibility.scores:
        legibility.tot_score = np.mean(list(legibility.scores.values()))
    return legibility.tot_score


def eval_predictability(other_agents, interaction, int_cost, max_cost):
    predictability = Score()
    for id in other_agents:
        if np.diff(interaction.int_idx[id]) > 1:
            cost_stg = interaction.int_t[id] + int_cost.tg[id]
            arg = int_cost.sg[id] - cost_stg
            arg = np.where(int_cost.rtg[id] > max_cost, -np.inf, arg)
            arg = np.clip(arg, -max_cost, 0)
            predictability.vals[id] = np.delete(np.exp(arg), 1, 0)
            predictability.scores[id] = np.max(predictability.vals[id][:, -1])
    if predictability.scores:
        predictability.tot_score = np.mean(list(predictability.scores.values()))
    return predictability.tot_score


def eval_nav_contrib(dt, other_agents, mpd, partials):
    eps = dict()
    nav_contrib = [np.nan]
    for id in other_agents:
        if np.all([a.size != 0 for a in mpd.params[id]]):
            dparams = [np.diff(np.pad(p, (1, 0), mode="edge")) for p in mpd.params[id]]
            dparams[1::2] = helper.wrap_to_pi(dparams[1::2])
            with np.errstate(invalid="ignore"):
                eps[id] = [
                    gaussian_filter(partial(*mpd.args[id]) * dp, sigma=3)
                    for partial, dp in zip(partials, dparams)
                ]
            if np.all(np.isfinite(eps[id])):
                cum_contrib = [np.trapz(e / dt, dx=dt) for e in eps[id]]
                if np.any(cum_contrib):
                    nav_contrib.append(sum(cum_contrib[:2]) / sum(cum_contrib))
    return np.nanmean(nav_contrib)


@dataclass
class Score:
    tot_score: float = np.nan
    vals: dict[int, np.ndarray] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)


@dataclass
class Interaction:
    int_idx: dict[int, list] = field(default_factory=dict)
    int_slice: dict[int, slice] = field(default_factory=dict)
    int_t: dict[int, np.ndarray] = field(default_factory=dict)
    int_line: dict[int, np.ndarray] = field(default_factory=dict)
    int_line_heading: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class IntCost:
    rg: dict[int, np.ndarray] = field(default_factory=dict)
    rtg: dict[int, np.ndarray] = field(default_factory=dict)
    tg: dict[int, np.ndarray] = field(default_factory=dict)
    sg: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class Mpd:
    params: dict[int, list] = field(default_factory=dict)
    args: dict[int, list] = field(default_factory=dict)
    vals: dict[int, np.ndarray] = field(default_factory=dict)


class Metric:
    def __init__(
        self,
        name,
        units,
        decimals,
        opt_func,
        update_func,
        scenarios,
        policies,
        val=np.inf,
        cnts=None,
    ):
        self.name = name
        self.units = units
        self.scale = 100 if self.units == "%" else 1
        self.decimals = decimals
        self.opt_func = opt_func
        self.scenarios = scenarios
        self.policies = policies
        self.update_func = update_func
        self.log = self.get_log(scenarios, policies, val, cnts)

    def __repr__(self):
        units = self.units.replace("%", "$\\%$")
        return f"{self.name} ({units})"

    def __call__(self, scenario, policy, iter, *args):
        self.log[scenario][policy][iter] = self.update_func(*args)

    def get_log(self, scenarios, policies, val, cnts):
        if cnts:
            return {s: {p: np.full(cnt, val) for p in policies} for s, cnt in zip(scenarios, cnts)}
        return {s: {p: val for p in policies} for s in scenarios}

    def compute_mean(self):
        self.mean = {
            s: {p: self.scale * np.nanmean(self.log[s][p]) for p in self.policies}
            for s in self.scenarios
        }

    def compute_std(self):
        self.std = {
            s: {p: self.scale * np.nanstd(self.log[s][p]) for p in self.policies}
            for s in self.scenarios
        }

    def get_opt(self):
        self.opt_val = {s: self.opt_func(list(self.mean[s].values())) for s in self.scenarios}

    def format_vals(self):
        self.formatted_vals = deepcopy(self.mean)
        for s, opt_val in zip(self.formatted_vals, self.opt_val.values()):
            for p, val in self.formatted_vals[s].items():
                self.formatted_vals[s][p] = f"{val:.{self.decimals}f}"
                if val == opt_val:
                    self.formatted_vals[s][p] = f"\033[1m{self.formatted_vals[s][p]}\033[0m"


class Eval:
    def __init__(self, config, trial_cnts):
        self.conf = config
        self.colors = {p: "" for p in self.conf.env.policies}
        self.init_metrics(self.conf.env.scenarios, self.conf.env.policies, trial_cnts)
        self.init_symbols()

    def init_metrics(self, scenarios, policies, trial_cnts):
        req_args = (scenarios, policies)
        self.metrics = {
            "extra_ttg": Metric(
                "Extra TTG", "%", 2, min, eval_extra_ttg, *req_args, cnts=trial_cnts
            ),
            "failure": Metric(
                "Failure Rate", "%", 0, min, eval_failure, *req_args, val=False, cnts=trial_cnts
            ),
            "efficiency": Metric(
                "Path Efficiency", "%", 2, max, eval_efficiency, *req_args, cnts=trial_cnts
            ),
            "irregularity": Metric(
                "Path Irregularity", "rad/m", 4, min, eval_irregularity, *req_args, cnts=trial_cnts
            ),
            "legibility": Metric(
                "Legibility", "%", 2, max, eval_legibility, *req_args, cnts=trial_cnts
            ),
            "predictability": Metric(
                "Predictability", "%", 2, max, eval_predictability, *req_args, cnts=trial_cnts
            ),
            "nav_contrib": Metric(
                "Navigation Contribution", "%", 2, max, eval_nav_contrib, *req_args, cnts=trial_cnts
            ),
        }

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

    def evaluate(self, iter, dt, ego_agent, scenario):
        self.colors[ego_agent.policy] = ego_agent.color
        req_args = (scenario, ego_agent.policy, iter)
        self.metrics["extra_ttg"](*req_args, ego_agent, self.conf.agent.goal_tol)
        self.metrics["failure"](*req_args, ego_agent)
        self.metrics["efficiency"](*req_args, ego_agent)
        self.metrics["irregularity"](*req_args, ego_agent)
        interaction = self.get_interactions(dt, ego_agent)
        int_cost = self.compute_int_costs(dt, ego_agent, ego_agent.other_agents, interaction)
        self.metrics["legibility"](*req_args, dt, ego_agent, interaction, int_cost, self.conf)
        self.metrics["predictability"](
            *req_args, ego_agent.other_agents, interaction, int_cost, self.conf.lpnav.max_cost
        )
        mpd = self.compute_mpd(ego_agent, interaction.int_slice)
        self.metrics["nav_contrib"](*req_args, dt, ego_agent.other_agents, mpd, self.mpd_partials)
        if self.conf.eval.show_nav_contrib_plot or self.conf.eval.save_nav_contrib_plot:
            self.make_nav_contrib_plot(env, mpd, eps)

    def get_interactions(self, dt, ego_agent):
        interaction = Interaction()
        interaction.int_line_heading = helper.wrap_to_pi(
            helper.angle(ego_agent.pos_log - ego_agent.goal)
        )
        for id, agent in ego_agent.other_agents.items():
            in_front = helper.in_front(
                agent.pos_log, interaction.int_line_heading, ego_agent.pos_log
            )
            inter_dist = helper.dist(ego_agent.pos_log, agent.pos_log)
            in_radius = inter_dist < self.conf.agent.sensing_dist
            col_width = ego_agent.radius + agent.radius
            rel_int_line = np.array([[0, -col_width], [0, col_width]])
            abs_int_line = helper.rotate(rel_int_line, interaction.int_line_heading)
            interaction.int_line[id] = abs_int_line + agent.pos_log
            feasible = (
                helper.cost_to_line(
                    ego_agent.pos_log,
                    ego_agent.max_speed,
                    interaction.int_line[id],
                    agent.vel_log,
                )
                < 1e2
            )
            is_interacting = in_front & in_radius & feasible
            if ~np.any(is_interacting) or dt * np.sum(is_interacting) < 1:
                start_idx, end_idx = (0, 0)
            else:
                start_idx = np.argmax(is_interacting)
                end_idx = np.nonzero(is_interacting)[0][-1]
            interaction.int_idx[id] = [start_idx, end_idx]
            interaction.int_slice[id] = slice(start_idx, end_idx + 1)
            interaction.int_t[id] = dt * np.arange(start_idx, end_idx + 1)
        return interaction

    def compute_int_costs(self, dt, ego_agent, other_agents, interaction):
        int_cost = IntCost()
        receding_steps = int(self.conf.lpnav.receding_horiz / dt)
        for id, agent in other_agents.items():  # Try and use a zip here
            init_receded_pos = (
                ego_agent.pos_log[0]
                - dt * np.arange(receding_steps + 1, 1, -1)[:, None] * ego_agent.vel_log[0]
            )
            receded_pos = np.concatenate((init_receded_pos, ego_agent.pos_log[:-receding_steps]))[
                interaction.int_slice[id]
            ]
            receded_line = (
                interaction.int_line[id][:, interaction.int_slice[id]]
                - agent.vel_log[interaction.int_slice[id]] * self.conf.lpnav.receding_horiz
            )
            receded_start_line = (
                interaction.int_line[id][:, interaction.int_slice[id]]
                - agent.vel_log[interaction.int_slice[id]] * interaction.int_t[id][:, None]
            )
            # scaled_speed = max(ego_agent.max_speed, agent.max_speed + 0.1)
            int_cost.rg[id] = helper.dynamic_pt_cost(
                receded_pos,
                ego_agent.max_speed,
                receded_line,
                interaction.int_line_heading[interaction.int_slice[id]],
                agent.vel_log[interaction.int_slice[id]],
            )
            int_cost.tg[id] = helper.dynamic_pt_cost(
                ego_agent.pos_log[interaction.int_slice[id]],
                ego_agent.max_speed,
                interaction.int_line[id][:, interaction.int_slice[id]],
                interaction.int_line_heading[interaction.int_slice[id]],
                agent.vel_log[interaction.int_slice[id]],
            )
            int_cost.rtg[id] = self.conf.lpnav.receding_horiz + int_cost.tg[id]
            int_cost.sg[id] = helper.dynamic_pt_cost(
                ego_agent.pos_log[interaction.int_idx[id][0]],
                ego_agent.max_speed,
                receded_start_line,
                interaction.int_line_heading[interaction.int_slice[id]],
                agent.vel_log[interaction.int_slice[id]],
            )
        return int_cost

    def compute_mpd(self, ego_agent, int_slice):
        mpd = Mpd()
        for id, agent in ego_agent.other_agents.items():
            mpd.params[id] = [
                ego_agent.speed_log[int_slice[id]],
                ego_agent.heading_log[int_slice[id]],
                agent.speed_log[int_slice[id]],
                agent.heading_log[int_slice[id]],
            ]
            mpd.args[id] = [
                ego_agent.pos_log[:, 0][int_slice[id]],
                ego_agent.pos_log[:, 1][int_slice[id]],
                agent.pos_log[:, 0][int_slice[id]],
                agent.pos_log[:, 1][int_slice[id]],
                *mpd.params[id],
            ]
            with np.errstate(invalid="ignore"):
                mpd.vals[id] = gaussian_filter(self.mpd(*mpd.args[id]), sigma=3)
        return mpd

    def make_nav_contrib_plot(self, env, mpd, eps):
        self.conf.animation.dark_bg and plt.style.use("dark_background")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(16, 4)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("Time (s)")
        for id, agent in env.ego_agent.other_agents.items():
            if mpd.vals[id].size != 0:
                t = np.linspace(0, self.conf.env.dt * len(mpd.vals[id]), len(mpd.vals[id]))
                ax1.plot(t, mpd.vals[id], label=r"$MPD(t)$", c=agent.color, lw=2)
                ax2.plot(
                    t,
                    eps[id][0],
                    c=env.ego_agent.color,
                    lw=2,
                    label=r"$\frac{\partial MPD}{\partial v_R}$",
                )
                ax2.plot(
                    t,
                    eps[id][1],
                    c=env.ego_agent.color,
                    lw=2,
                    ls="--",
                    label=r"$\frac{\partial MPD}{\partial \theta_R}$",
                )
                ax2.plot(
                    t,
                    eps[id][2],
                    c=agent.color,
                    lw=2,
                    label=r"$\frac{\partial MPD}{\partial v_H}$",
                )
                ax2.plot(
                    t,
                    eps[id][3],
                    c=agent.color,
                    lw=2,
                    ls="--",
                    label=r"$\frac{\partial MPD}{\partial \theta_H}$",
                )
                ax3.plot(t, env.dt * np.cumsum(eps[id][0]), c=env.ego_agent.color)
                ax3.plot(t, env.dt * np.cumsum(eps[id][1]), c=env.ego_agent.color, ls="--", lw=2)
                ax3.plot(t, env.dt * np.cumsum(eps[id][2]), c="gray", lw=2)
                ax3.plot(t, env.dt * np.cumsum(eps[id][3]), c="gray", ls="--", lw=2)
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

    def make_tbl(self):
        for s in self.conf.env.scenarios:
            tbl = PrettyTable()
            tbl.title = s
            if s in ("random", "circle"):
                tbl.title += f"_{self.conf.env.num_of_agents}_agents"
            if self.conf.env.homogeneous:
                tbl.title += "_homogeneous"
            if s == "random":
                tbl.title += f"_{self.conf.env.random_scenarios}_iters"
            tbl.add_column("Policies", self.conf.env.policies)
            for m in [v for k, v in self.metrics.items() if k in self.conf.eval.individual_metrics]:
                tbl.add_column(m.name, list(m.formatted_vals[s].values()))
            self.conf.eval.show_tbl and print(tbl)
            self.conf.eval.save_tbl and self.save_tbl(tbl)
        for m in self.conf.eval.individual_metrics:
            tbl = PrettyTable()
            tbl.title = m
            tbl.add_column("Policies", self.conf.env.policies)
            for s, vals in self.metrics[m].formatted_vals.items():
                tbl.add_column(s, list(vals.values()))
            self.conf.eval.show_tbl and print(tbl)
            self.conf.eval.save_tbl and self.save_tbl(tbl)

    def save_tbl(self, tbl):
        indent = "  "
        newline = "\\\\"
        rows = [
            f"\\begin{{tabularx}}{{\\textwidth}}{{@{{}}X*{{{len(tbl.field_names) - 1}}}{{Y}}@{{}}}}"
        ]
        rows += [f"{indent}\\toprule"]
        rows += [f"{indent}{' & '.join(tbl.field_names)} {newline}"]
        rows += [f"{indent}\\midrule"]
        tex_rows = [
            [v.replace("\033[1m", "\\textbf{").replace("\033[0m", "}") for v in row]
            for row in tbl.rows
        ]
        rows += [f"{indent}{' & '.join(row)} {newline}" for row in tex_rows]
        rows += [f"{indent}\\bottomrule"]
        rows += ["\\end{tabularx}"]
        contents = "\n".join(rows)
        os.makedirs(self.conf.eval.tbl_dir, exist_ok=True)
        fname = os.path.join(self.conf.eval.tbl_dir, f"{tbl.title}.tex")
        with open(fname, "w") as f:
            f.write(contents)

    def get_summary(self):
        for metric in self.metrics.values():
            metric.compute_mean()
            metric.compute_std()
            metric.get_opt()
            metric.format_vals()
        (self.conf.eval.show_tbl or self.conf.eval.save_tbl) and self.make_tbl()
