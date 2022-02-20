from copy import deepcopy
import logging
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import sympy as smp
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter
from legitable.policies.agent import Agent
from utils import helper

from legitable.utils.opt_traj import get_opt_traj


def eval_extra_ttg(ego_agent, goal_tol):
    if hasattr(ego_agent, "time_to_goal"):
        opt_ttg = (helper.dist(ego_agent.start, ego_agent.goal) - goal_tol) / ego_agent.max_speed
        return (ego_agent.time_to_goal - opt_ttg) / opt_ttg
    return np.nan


def eval_failure(ego_agent):
    return False if hasattr(ego_agent, "time_to_goal") else True


def eval_efficiency(ego_agent):
    if hasattr(ego_agent, "time_to_goal"):
        path_len = np.sum(np.linalg.norm(np.diff(ego_agent.pos_log, axis=0), axis=-1))
        opt_path = helper.dist(ego_agent.start, ego_agent.goal) - ego_agent.goal_tol
        return 0 if not path_len else opt_path / path_len
    return np.nan


def eval_irregularity(ego_agent):
    return np.mean(np.abs(ego_agent.heading_log - helper.angle(ego_agent.goal - ego_agent.pos_log)))


def eval_legibility(dt, ego_agent, interaction, goal_inference, config):
    leg_scores = {}
    for id, agent in ego_agent.other_agents.items():
        if np.diff(interaction.int_idx[id]) > 1:
            t_discount = interaction.int_t[id][::-1]
            num = np.trapz(t_discount * goal_inference[id], dx=dt)
            den = np.trapz(t_discount, dx=dt)
            leg_scores[id] = np.delete(num / den, 1)[interaction.passing_idx[id]]
            if config.eval.normalize_lp:
                min_leg, max_leg = get_opt_traj(
                    interaction.int_idx[id],
                    interaction.int_slice[id],
                    interaction.passing_idx[id],
                    dt,
                    ego_agent,
                    agent,
                    config.lpnav.receding_horiz,
                    config.lpnav.subgoal_priors,
                )
                if min_leg is not None and max_leg is not None:
                    leg_scores[id] = (leg_scores[id] - min_leg) / (max_leg - min_leg)
                    assert 0 <= leg_scores[id] <= 1.1
    return np.mean(list(leg_scores.values())) if leg_scores else np.nan


def eval_predictability(other_agents, interaction, traj_inference):
    pred_scores = {}
    for id in other_agents:
        if np.diff(interaction.int_idx[id]) > 1:
            passing_sides = np.delete(traj_inference[id], 1, 0)
            pred_scores[id] = np.max(passing_sides[:, -1])
    return np.mean(list(pred_scores.values())) if pred_scores else np.nan


def eval_nav_contrib(dt, other_agents, mpd, partials):
    eps = {}
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


def eval_pass_uncertainty(steps, interaction, goal_inf):
    vals = np.full(int(steps), False)
    for id, inf in goal_inf.items():
        pass_idx = 0 if interaction.passing_idx[id] == 0 else 2
        true_pass_inf = inf[pass_idx]
        other_pass_inf = inf[2 if pass_idx == 0 else 0]
        # vals[interaction.int_slice[id]] |= other_pass_inf < 0.5
        vals[interaction.int_slice[id]] |= true_pass_inf < other_pass_inf
    return np.sum(vals) / len(vals)


def eval_min_pass_inf(ego_agent, interaction, goal_inf):
    pass_inf = np.full(len(ego_agent.pos_log), np.inf)
    for id in goal_inf:
        sl = interaction.int_slice[id]
        if sl.stop - sl.start > 1:
            sliced_inf = goal_inf[id][0 if interaction.passing_idx[id] == 0 else 2]
            pass_inf[sl] = np.where(sliced_inf < pass_inf[sl], sliced_inf, pass_inf[sl])
    return np.min(pass_inf) if np.any(np.isfinite(pass_inf)) else np.nan


def get_int_costs(dt, ego_agent, other_agents, interaction, receding_horiz):
    int_cost = IntCost()
    receding_steps = int(receding_horiz / dt)
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
            - agent.vel_log[interaction.int_slice[id]] * receding_horiz
        )
        receded_start_line = (
            interaction.int_line[id][:, interaction.int_slice[id]]
            - agent.vel_log[interaction.int_slice[id]] * interaction.int_t[id][:, None]
        )
        scaled_speed = max(ego_agent.max_speed, agent.max_speed + 0.1)
        int_cost.rg[id] = helper.dynamic_pt_cost(
            receded_pos,
            scaled_speed,
            receded_line,
            interaction.int_line_heading[interaction.int_slice[id]],
            agent.vel_log[interaction.int_slice[id]],
        )
        int_cost.tg[id] = helper.dynamic_pt_cost(
            ego_agent.pos_log[interaction.int_slice[id]],
            scaled_speed,
            interaction.int_line[id][:, interaction.int_slice[id]],
            interaction.int_line_heading[interaction.int_slice[id]],
            agent.vel_log[interaction.int_slice[id]],
        )
        int_cost.rtg[id] = receding_horiz + int_cost.tg[id]
        int_cost.sg[id] = helper.dynamic_pt_cost(
            ego_agent.pos_log[interaction.int_idx[id][0]],
            scaled_speed,
            receded_start_line,
            interaction.int_line_heading[interaction.int_slice[id]],
            agent.vel_log[interaction.int_slice[id]],
        )
    return int_cost


def get_interactions(dt, ego_agent, sensing_dist):
    interaction = Interaction()
    interaction.int_line_heading = helper.wrap_to_pi(
        helper.angle(ego_agent.pos_log - ego_agent.goal)
    )
    for id, agent in ego_agent.other_agents.items():
        in_front = helper.in_front(agent.pos_log, interaction.int_line_heading, ego_agent.pos_log)
        a_in_front = helper.in_front(agent.pos_log, agent.heading_log, ego_agent.pos_log)
        inter_dist = helper.dist(ego_agent.pos_log, agent.pos_log)
        in_radius = inter_dist < sensing_dist
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
        is_interacting = in_front & in_radius & feasible & a_in_front
        if ~np.any(is_interacting) or dt * np.sum(is_interacting) < 1:
            start_idx, end_idx = (0, 0)
        else:
            interaction.agents[id] = agent
            start_idx = np.argmax(is_interacting)
            end_idx = np.nonzero(is_interacting)[0][-1]
            end_idx = np.argmin(np.abs(np.cross(np.diff(interaction.int_line[id], axis=0)[0], ego_agent.pos_log - interaction.int_line[id][0,:,:], axis=1))) - 1
            th = helper.angle(np.diff(interaction.int_line[id][:, end_idx], axis=0))
            right = helper.in_front(agent.pos_log[end_idx], th, ego_agent.pos_log[end_idx])
            interaction.passing_idx[id] = 1 if right else 0
        interaction.int_idx[id] = [start_idx, end_idx]
        interaction.int_slice[id] = slice(start_idx, end_idx + 1)
        interaction.int_t[id] = dt * np.arange(start_idx, end_idx + 1)
    return interaction


def get_goal_inference(other_agents, interaction, int_cost, priors):
    goal_inference = {}
    for id in other_agents:
        if np.diff(interaction.int_idx[id]) > 1:
            arg = int_cost.rg[id] - int_cost.rtg[id]
            goal_inference[id] = np.exp(arg) * np.expand_dims(priors, axis=-1)
            goal_inference[id] /= np.sum(goal_inference[id], axis=0)
    return goal_inference


def get_traj_inference(other_agents, interaction, int_cost):
    traj_inference = {}
    for id in other_agents:
        if np.diff(interaction.int_idx[id]) > 1:
            cost_stg = interaction.int_t[id] + int_cost.tg[id]
            traj_inference[id] = np.exp(int_cost.sg[id] - cost_stg)
    return traj_inference


def get_mpd(ego_agent, int_slice, mpd_fn):
    mpd = Mpd()
    for id, agent in ego_agent.other_agents.items():
        mpd.params[id] = [
            ego_agent.speed_log[int_slice[id]],
            ego_agent.heading_log[int_slice[id]],
            agent.speed_log[int_slice[id]],
            agent.heading_log[int_slice[id]],
        ]
        mpd.args[id] = [
            *ego_agent.pos_log[int_slice[id]].T,
            *agent.pos_log[int_slice[id]].T,
            *mpd.params[id],
        ]
        with np.errstate(invalid="ignore"):
            mpd.vals[id] = gaussian_filter(mpd_fn(*mpd.args[id]), sigma=3)
    return mpd


@dataclass
class Score:
    tot_score: float = np.nan
    vals: dict[int, np.ndarray] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)


@dataclass
class Interaction:
    agents: dict[int, Agent] = field(default_factory=dict)
    int_idx: dict[int, list] = field(default_factory=dict)
    int_slice: dict[int, slice] = field(default_factory=dict)
    int_t: dict[int, np.ndarray] = field(default_factory=dict)
    int_line: dict[int, np.ndarray] = field(default_factory=dict)
    int_line_heading: dict[int, np.ndarray] = field(default_factory=dict)
    passing_idx: dict[int, int] = field(default_factory=dict)


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
        num_of_agents_lst,
        trial_cnts,
        val=np.nan,
    ):
        self.name = name
        self.units = units
        self.scale = 100 if self.units == "%" else 1
        self.decimals = decimals
        self.opt_func = opt_func
        self.scenarios = scenarios
        self.policies = policies
        self.update_func = update_func
        self.log = self.get_log(self.scenarios, self.policies, num_of_agents_lst, val, trial_cnts)

    def __repr__(self):
        units = self.units.replace("%", "$\\%$")
        return f"{self.name} ({units})"

    def __call__(self, scenario, policy, iter, n, *args):
        self.log[scenario][n][policy][iter] = self.update_func(*args)

    def get_log(self, scenarios, policies, num_of_agents_lst, val, trial_cnts):
        return {
            s: {n: {p: np.full(cnt, val) for p in policies} for n in num_of_agents}
            for s, cnt, num_of_agents in zip(scenarios, trial_cnts, num_of_agents_lst)
        }

    def compute_mean(self):
        self.mean = {
            s: {
                n: {p: self.scale * np.nanmean(self.log[s][n][p]) for p in self.log[s][n]}
                for n in self.log[s]
            }
            for s in self.log
        }

    def compute_std(self):
        self.std = {
            s: {
                n: {p: self.scale * np.nanstd(self.log[s][n][p]) for p in self.log[s][n]}
                for n in self.log[s]
            }
            for s in self.log
        }

    def get_opt(self):
        self.opt_val = {
            s: {n: self.opt_func(list(self.mean[s][n].values())) for n in self.mean[s]}
            for s in self.scenarios
        }

    def format_vals(self):
        self.formatted_vals = deepcopy(self.mean)
        for s in self.formatted_vals:
            for n, opt_val in self.opt_val[s].items():
                for p, val in self.formatted_vals[s][n].items():
                    self.formatted_vals[s][n][p] = f"{val:.{self.decimals}f}"
                    if val == opt_val:
                        self.formatted_vals[s][n][p] = f"<{self.formatted_vals[s][n][p]}>"
        self.formatted_vals = {
            s: {
                p: " / ".join([self.formatted_vals[s][n][p] for n in self.formatted_vals[s]])
                for p in self.policies
            }
            for s in self.formatted_vals
        }


class Feature:
    def __init__(self, update_func, scenarios, policies, num_of_agents_lst):
        self.update_func = update_func
        self.log = self.get_log(scenarios, policies, num_of_agents_lst)

    def __call__(self, scenario, policy, n, *args):
        res = self.update_func(*args)
        if res is not None:
            self.log[scenario][n][policy].append(res)

    def get_log(self, scenarios, policies, num_of_agents_lst):
        return {
            s: {n: {p: [] for p in policies} for n in num_of_agents}
            for s, num_of_agents in zip(scenarios, num_of_agents_lst)
        }


class Eval:
    def __init__(self, config, num_of_agents_lst, trial_cnts):
        self.conf = config
        self.colors = {p: "" for p in self.conf.env.policies}
        self.policy_dict = {
            "lpnav": "LPNav",
            "social_momentum": "SM",
            "sa_cadrl": "SA-CADRL",
            "ga3c_cadrl": "GA3C-CADRL",
            "rvo": "RVO",
            "sfm": "SFM",
        }
        self.init_metrics(
            self.conf.env.scenarios, self.conf.env.policies, num_of_agents_lst, trial_cnts
        )
        self.init_feats(self.conf.env.scenarios, self.conf.env.policies, num_of_agents_lst)
        self.init_mpd_symbols()
        self.logger = logging.getLogger(__name__)

    def init_metrics(self, scenarios, policies, num_of_agents_lst, trial_cnts):
        args = (scenarios, policies, num_of_agents_lst, trial_cnts)
        self.metrics = {
            "extra_ttg": Metric("Extra TTG", "%", 2, min, eval_extra_ttg, *args),
            "failure": Metric("Failure Rate", "%", 0, min, eval_failure, *args, val=False),
            "efficiency": Metric("Path Efficiency", "%", 2, max, eval_efficiency, *args),
            "irregularity": Metric("Path Irregularity", "rad/m", 4, min, eval_irregularity, *args),
            "legibility": Metric("Legibility", "%", 2, max, eval_legibility, *args),
            "predictability": Metric("Predictability", "%", 2, max, eval_predictability, *args),
            "nav_contrib": Metric("Navigation Contribution", "%", 2, max, eval_nav_contrib, *args),
            "pass_uncertainty": Metric("Passing Uncertainty", "%", 2, min, eval_pass_uncertainty, *args),
            "min_pass_inf": Metric("Minimum Passing Inference", "%", 2, max, eval_min_pass_inf, *args)
        }

    def init_feats(self, scenarios, policies, num_of_agents_lst):
        args = (scenarios, policies, num_of_agents_lst)
        self.feats = {
            "int_cost": Feature(get_int_costs, *args),
            "interaction": Feature(get_interactions, *args),
            "goal_inference": Feature(get_goal_inference, *args),
            "traj_inference": Feature(get_traj_inference, *args),
            "mpd": Feature(get_mpd, *args),
        }

    def init_mpd_symbols(self):
        prx, pry, phx, phy, vr, thr, vh, thh = smp.symbols("prx pry phx phy vr thr vh thh")
        dvx = vh * smp.cos(thh) - vr * smp.cos(thr)
        dvy = vr * smp.sin(thr) - vh * smp.sin(thh)
        dpx = phx - prx
        dpy = phy - pry
        args = (prx, pry, phx, phy, vr, thr, vh, thh)
        mpd = smp.sqrt((dvx * dpy + dvy * dpx) ** 2 / (dvx ** 2 + dvy ** 2))
        self.mpd_fn = smp.lambdify(args, mpd)
        self.mpd_partials = [smp.lambdify(args, smp.diff(mpd, p)) for p in [vr, thr, vh, thh]]

    def evaluate(self, iter, dt, ego_agent, scenario, n):
        self.colors[ego_agent.policy] = ego_agent.color
        feat_args = (scenario, ego_agent.policy, n)
        metric_args = (scenario, ego_agent.policy, iter, n)
        self.metrics["extra_ttg"](*metric_args, ego_agent, self.conf.agent.goal_tol)
        self.metrics["failure"](*metric_args, ego_agent)
        self.metrics["efficiency"](*metric_args, ego_agent)
        self.metrics["irregularity"](*metric_args, ego_agent)
        self.feats["interaction"](*feat_args, dt, ego_agent, self.conf.agent.sensing_dist)
        self.feats["int_cost"](
            *feat_args,
            dt,
            ego_agent,
            ego_agent.other_agents,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.conf.lpnav.receding_horiz,
        )
        self.feats["goal_inference"](
            *feat_args,
            ego_agent.other_agents,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.feats["int_cost"].log[scenario][n][ego_agent.policy][iter],
            self.conf.lpnav.subgoal_priors,
        )
        self.metrics["legibility"](
            *metric_args,
            dt,
            ego_agent,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.feats["goal_inference"].log[scenario][n][ego_agent.policy][iter],
            self.conf,
        )
        self.feats["traj_inference"](
            *feat_args,
            ego_agent.other_agents,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.feats["int_cost"].log[scenario][n][ego_agent.policy][iter],
        )
        self.metrics["predictability"](
            *metric_args,
            ego_agent.other_agents,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.feats["traj_inference"].log[scenario][n][ego_agent.policy][iter],
        )
        self.feats["mpd"](
            *feat_args,
            ego_agent,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter].int_slice,
            self.mpd_fn,
        )
        self.metrics["nav_contrib"](
            *metric_args,
            dt,
            ego_agent.other_agents,
            self.feats["mpd"].log[scenario][n][ego_agent.policy][iter],
            self.mpd_partials,
        )
        self.metrics["pass_uncertainty"](
            *metric_args,
            ego_agent.pos_log.shape[0],
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.feats["goal_inference"].log[scenario][n][ego_agent.policy][iter],
        )
        self.metrics["min_pass_inf"](
            *metric_args,
            ego_agent,
            self.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
            self.feats["goal_inference"].log[scenario][n][ego_agent.policy][iter],
        )
        if self.conf.eval.show_nav_contrib_plot or self.conf.eval.save_nav_contrib_plot:
            self.make_nav_contrib_plot(env, mpd, eps)

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
                tbl.title += f"_{'_'.join(map(str, self.conf.env.num_of_agents))}_agents"
            if self.conf.env.homogeneous:
                tbl.title += "_homogeneous"
            if s == "random":
                tbl.title += f"_{self.conf.env.random_scenarios}_iters"
            tbl.add_column("Policies", [self.policy_dict.get(p, p) for p in self.conf.env.policies])
            for m in [v for k, v in self.metrics.items() if k in self.conf.eval.metrics]:
                tbl.add_column(f"{m.name} ({m.units})", list(m.formatted_vals[s].values()))
            self.conf.eval.show_tbl and self.logger.info('\n' + str(tbl))
            self.conf.eval.save_tbl and self.save_tbl(tbl)
        for m in self.conf.eval.individual_metrics:
            tbl = PrettyTable()
            tbl.title = m
            tbl.add_column("Policies", self.conf.env.policies)
            for s, vals in self.metrics[m].formatted_vals.items():
                tbl.add_column(s, list(vals.values()))
            self.conf.eval.show_tbl and self.logger.info('\n' + str(tbl))
            self.conf.eval.save_tbl and self.save_tbl(tbl)

    def save_tbl(self, tbl):
        tbl.field_names = [n.replace('%', "\\%") for n in tbl.field_names]
        rows = [
            f"\\begin{{tabularx}}{{\\textwidth}}{{@{{}}X*{{{len(tbl.field_names) - 1}}}{{Y}}@{{}}}}"
        ]
        indent = "  "
        newline = "\\\\"
        rows += [f"{indent}\\toprule"]
        rows += [f"{indent}{' & '.join(tbl.field_names)} {newline}"]
        rows += [f"{indent}\\midrule"]
        tex_rows = [
            [v.replace("<", "\\textbf{").replace(">", "}").replace("_", "\\_").replace("%", "\\%") for v in row]
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
        for metric in [m for m in self.metrics.values() if m.opt_func]:
            metric.compute_mean()
            metric.compute_std()
            metric.get_opt()
            metric.format_vals()
        (self.conf.eval.show_tbl or self.conf.eval.save_tbl) and self.make_tbl()
