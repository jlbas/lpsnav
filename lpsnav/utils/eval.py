import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from utils import helper
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as smp
from scipy.ndimage import gaussian_filter


prx, pry, phx, phy, vr, thr, vh, thh = smp.symbols("prx pry phx phy vr thr vh thh")
dvx = vh * smp.cos(thh) - vr * smp.cos(thr)
dvy = vr * smp.sin(thr) - vh * smp.sin(thh)
dpx = phx - prx
dpy = phy - pry
args = (prx, pry, phx, phy, vr, thr, vh, thh)
mpd = smp.sqrt((dvx * dpy + dvy * dpx) ** 2 / (dvx ** 2 + dvy ** 2))
mpd_fn = smp.lambdify(args, mpd)


def get_interactions(conf, env):
    interactions = {}
    goal_vec = env.logs[env.ego_id].pos - env.agents[env.ego_id].goal
    line_heading = helper.wrap_to_pi(helper.angle(goal_vec))
    for k in list(env.agents)[1:]:
        inter_dist = helper.dist(env.logs[env.ego_id].pos, env.logs[k].pos)
        in_radius = inter_dist < conf["agent"]["sensing_dist"]

        in_front = helper.in_front(env.logs[k].pos, line_heading, env.logs[env.ego_id].pos)
        in_horiz = (
            helper.cost_to_line_th(
                env.logs[env.ego_id].pos,
                env.logs[env.ego_id].speed,
                env.logs[k].pos,
                env.logs[k].vel,
                line_heading,
            )
            < conf["agent"]["lpsnav"]["sensing_horiz"]
        )
        is_interacting = in_front & in_radius & in_horiz
        start_idx = np.argmax(is_interacting)
        int_heading = np.full(line_heading.shape, line_heading[start_idx])

        in_front = helper.in_front(env.logs[k].pos, int_heading, env.logs[env.ego_id].pos)
        in_horiz = (
            helper.cost_to_line_th(
                env.logs[env.ego_id].pos,
                env.logs[env.ego_id].speed,
                env.logs[k].pos,
                env.logs[k].vel,
                int_heading,
            )
            < conf["agent"]["lpsnav"]["sensing_horiz"]
        )
        is_interacting = in_front & in_radius & in_horiz
        rem = is_interacting[start_idx:]
        end_idx = start_idx + (len(rem) - 1 if np.all(rem) else np.argmin(rem))

        col_width = env.agents[env.ego_id].radius + env.agents[k].radius
        rel_line = np.array([[0, -col_width], [0, col_width]])
        abs_line = helper.rotate(rel_line, int_heading)
        line = abs_line + env.logs[k].pos
        if env.dt * (end_idx - start_idx) > 1:
            th = helper.angle(np.diff(line[:, end_idx], axis=0))
            right = helper.in_front(
                env.logs[k].pos[end_idx], th, env.logs[env.ego_id].pos[end_idx]
            )
            pass_idx = 1 if right else 0
            sl = slice(start_idx, end_idx)
            time = env.dt * np.arange(start_idx, end_idx)
            interactions[k] = Interaction(start_idx, end_idx, sl, time, line, int_heading, pass_idx)
    return interactions


def get_int_costs(conf, env, interactions):
    int_costs = {}
    receding_steps = int(conf["agent"]["lpsnav"]["receding_horiz"] / env.dt)
    for k, interaction in interactions.items():
        init_receded_pos = (
            env.logs[env.ego_id].pos[0]
            - env.dt * np.arange(receding_steps + 1, 1, -1)[:, None] * env.logs[env.ego_id].vel[0]
        )
        receded_pos = np.concatenate(
            (init_receded_pos, env.logs[env.ego_id].pos[:-receding_steps])
        )[interaction.slice]
        receded_line = (
            interaction.line[:, interaction.slice]
            - env.logs[k].vel[interaction.slice] * conf["agent"]["lpsnav"]["receding_horiz"]
        )
        receded_start_line = (
            interaction.line[:, interaction.slice]
            - env.logs[k].vel[interaction.slice] * interaction.time[:, None]
        )
        scaled_speed = max(env.agents[env.ego_id].max_speed, env.agents[k].max_speed + 0.1)
        cost_rg = helper.dynamic_pt_cost(
            receded_pos,
            scaled_speed,
            receded_line,
            interaction.line_heading[interaction.slice],
            env.logs[k].vel[interaction.slice],
        )
        cost_tg = helper.dynamic_pt_cost(
            env.logs[env.ego_id].pos[interaction.slice],
            scaled_speed,
            interaction.line[:, interaction.slice],
            interaction.line_heading[interaction.slice],
            env.logs[k].vel[interaction.slice],
        )
        cost_rtg = conf["agent"]["lpsnav"]["receding_horiz"] + cost_tg
        cost_sg = helper.dynamic_pt_cost(
            env.logs[env.ego_id].pos[interaction.start_idx],
            scaled_speed,
            receded_start_line,
            interaction.line_heading[interaction.slice],
            env.logs[k].vel[interaction.slice],
        )
        int_costs[k] = IntCost(cost_rg, cost_rtg, cost_tg, cost_sg)
    return int_costs


def get_goal_inference(conf, _env, int_costs):
    goal_infs = {}
    for k, int_cost in int_costs.items():
        arg = int_cost.rg - int_cost.rtg
        goal_infs[k] = np.exp(arg) * np.expand_dims(
            conf["agent"]["lpsnav"]["subgoal_priors"], axis=-1
        )
        goal_infs[k] /= np.sum(goal_infs[k], axis=0)
    return goal_infs


def get_traj_inference(_conf, _env, interactions, int_costs):
    traj_inference = {}
    for (k, interaction), int_cost in zip(interactions.items(), int_costs.values()):
        if interaction.end_idx - interaction.start_idx > 1:
            cost_stg = interaction.time + int_cost.tg
            traj_inference[k] = np.exp(int_cost.sg - cost_stg)
    return traj_inference


def get_mpd(_conf, env, interactions):
    mpds = {}
    for k, interaction in interactions.items():
        with np.errstate(invalid="ignore"):
            start = interaction.slice.start
            tcross = start + np.argmin(helper.dist(env.logs[env.ego_id].pos[start:], env.logs[k].pos[start:]))
            sl = slice(start, tcross + 1)
            mpd_vals = gaussian_filter(mpd_fn(
                *env.logs[env.ego_id].pos[sl].T,
                *env.logs[k].pos[sl].T,
                env.logs[env.ego_id].speed[sl],
                env.logs[env.ego_id].heading[sl],
                env.logs[k].speed[sl],
                env.logs[k].heading[sl],
            ), sigma=2)
            mpds[k] = Mpd(sl, env.time_log[sl], mpd_vals)
    return mpds


def eval_extra_ttg(conf, env, id=None):
    id = env.ego_id if id is None else id
    if hasattr(env.agents[id], "ttg"):
        if env.agents[id].ttg == 0:
            return 0
        goal_dist = helper.dist(env.agents[id].start, env.agents[id].goal) - conf["agent"]["goal_tol"]
        opt_ttg = goal_dist / env.agents[id].max_speed
        return (env.agents[id].ttg - opt_ttg) / opt_ttg
    return np.nan


def eval_others_extra_ttg(conf, env):
    return np.nanmean([eval_extra_ttg(conf, env, k) for k in env.agents if k != env.ego_id])


def eval_extra_dist(conf, env, id=None):
    id = env.ego_id if id is None else id
    if hasattr(env.agents[id], "ttg"):
        goal_dist = helper.dist(env.agents[id].start, env.agents[id].goal)
        goal_dist = max(0, goal_dist - conf["agent"]["goal_tol"])
        path_len = helper.path_len(env.logs[id].pos[:int(np.around(env.agents[id].ttg, 2) / env.dt)+2])
        return path_len - goal_dist
    return np.nan


def eval_others_extra_dist(conf, env):
    return np.nanmean([eval_extra_dist(conf, env, k) for k in env.agents if k != env.ego_id])


def eval_failure(_conf, env):
    return 0 if hasattr(env.agents[env.ego_id], "ttg") else 1


def eval_others_failure(_conf, env):
    return np.mean([not hasattr(env.agents[k], "ttg") for k in env.agents if k != env.ego_id])


def eval_efficiency(conf, env, id=None):
    id = env.ego_id if id is None else id
    if hasattr(env.agents[id], "ttg"):
        path_len = helper.path_len(env.logs[id].pos)
        goal_dist = helper.dist(env.agents[id].start, env.agents[id].goal)
        goal_dist = max(0, goal_dist - conf["agent"]["goal_tol"])
        opt_path = goal_dist - env.agents[id].goal_tol
        return 0 if not path_len else opt_path / path_len
    return np.nan


def eval_others_efficiency(conf, env):
    return np.nanmean([eval_efficiency(conf, env, k) for k in env.agents if k != env.ego_id])


def eval_irregularity(_conf, env, id=None):
    id = env.ego_id if id is None else id
    goal_heading = helper.angle(env.agents[id].goal - env.logs[id].pos)
    return np.mean(np.abs(env.logs[id].heading - goal_heading))


def eval_others_irregularity(conf, env):
    return np.nanmean([eval_irregularity(conf, env, k) for k in env.agents if k != env.ego_id])


def eval_legibility(_conf, env, interactions, goal_inferences):
    leg_scores = {}
    for (k, interaction), goal_inf in zip(interactions.items(), goal_inferences.values()):
        t_discount = interaction.time[::-1]
        num = np.trapz(t_discount * goal_inf, dx=env.dt)
        den = np.trapz(t_discount, dx=env.dt)
        leg_scores[k] = np.delete(num / den, 1)[interaction.passing_idx]
    return np.nan if not leg_scores else np.mean(list(leg_scores.values()))


def eval_predictability(_conf, _env, traj_inferences):
    pred_scores = {}
    for k, traj_inf in traj_inferences.items():
        passing_sides = np.delete(traj_inf, 1, 0)
        pred_scores[k] = np.max(passing_sides[:, -1])
    return np.nan if not pred_scores else np.mean(list(pred_scores.values()))


def eval_min_pass_inf(_conf, env, interactions, goal_inferences):
    pass_inf = np.full(len(env.logs[env.ego_id].pos), np.inf)
    for interaction, goal_inf in zip(interactions.values(), goal_inferences.values()):
        sl = interaction.slice
        if sl.stop - sl.start > 1:
            sliced_inf = goal_inf[0 if interaction.passing_idx == 0 else 2]
            pass_inf[sl] = np.where(sliced_inf < pass_inf[sl], sliced_inf, pass_inf[sl])
    return np.min(pass_inf) if np.any(np.isfinite(pass_inf)) else np.nan


def eval_avg_min_pass_inf(_conf, env, interactions, goal_inferences):
    pass_inf = np.full(len(env.logs[env.ego_id].pos), np.inf)
    for interaction, goal_inf in zip(interactions.values(), goal_inferences.values()):
        sl = interaction.slice
        if sl.stop - sl.start > 1:
            sliced_inf = goal_inf[0 if interaction.passing_idx == 0 else 2]
            pass_inf[sl] = np.where(sliced_inf < pass_inf[sl], sliced_inf, pass_inf[sl])
    pass_inf = np.where(np.isfinite(pass_inf), pass_inf, np.nan)
    return np.nanmean(pass_inf) if np.any(np.isfinite(pass_inf)) else np.nan


def eval_mpd(_conf, _env, mpd):
    return {k: [mpd.val.time[k], mpd.val.vals[k]] for k in mpd.val.time}


def eval_tracked_extra_ttg(conf, env):
    if hasattr(env.agents[env.ego_id], "ttg"):
        goal_dist = helper.dist(env.agents[env.ego_id].start, env.agents[env.ego_id].goal)
        goal_dist = max(0, goal_dist - conf["agent"]["goal_tol"])
        opt_ttg = goal_dist / env.agents[env.ego_id].max_speed
        tracked_rem_goal_dist = helper.dist(env.logs[env.ego_id].pos, env.agents[env.ego_id].goal)
        tracked_rem_goal_dist = np.maximum(0, tracked_rem_goal_dist - conf["agent"]["goal_tol"])
        tracked_ttg = env.time_log + tracked_rem_goal_dist / env.agents[env.ego_id].max_speed
        tracked_extra_ttg = (tracked_ttg - opt_ttg) / opt_ttg
        return {env.ego_id: [env.time_log, tracked_extra_ttg]}
    return {}


def eval_tracked_extra_dist(conf, env):
    if hasattr(env.agents[env.ego_id], "ttg"):
        goal_dist = helper.dist(env.agents[env.ego_id].start, env.agents[env.ego_id].goal)
        goal_dist = max(0, goal_dist - conf["agent"]["goal_tol"])
        tracked_path_len = np.cumsum(np.linalg.norm(np.diff(env.logs[env.ego_id].pos, axis=0), axis=-1))
        tracked_path_len = np.insert(tracked_path_len, 0, 0)
        tracked_rem_goal_dist = helper.dist(env.logs[env.ego_id].pos, env.agents[env.ego_id].goal)
        tracked_rem_goal_dist = np.maximum(0, tracked_rem_goal_dist - conf["agent"]["goal_tol"])
        tracked_extra_dist = tracked_path_len + tracked_rem_goal_dist - goal_dist
        return {env.ego_id: [env.time_log, tracked_extra_dist]}
    return {}


def eval_pass_inf(_conf, env, interactions, goal_inferences):
    pass_inf = {}
    for (k, interaction), goal_inf in zip(interactions.items(), goal_inferences.values()):
        pass_inf[k] = [interaction.time, goal_inf[0 if interaction.passing_idx == 0 else 2]]
    return pass_inf


@dataclass
class Interaction:
    start_idx: int
    end_idx: int
    slice: slice
    time: np.ndarray
    line: np.ndarray
    line_heading: np.ndarray
    passing_idx: int


@dataclass
class IntCost:
    rg: np.ndarray
    rtg: np.ndarray
    tg: np.ndarray
    sg: np.ndarray


@dataclass
class Mpd:
    slice: slice
    time: np.ndarray
    vals: np.ndarray


@dataclass
class Feature:
    update_func: Callable
    f_args: list[str] = field(default_factory=list)
    val: dict[int, object] = field(default_factory=dict)

    def __call__(self, *args):
        return self.update_func(*args)


@dataclass
class Metric:
    name: str
    units: str
    decimals: int
    opt_func: Callable
    update_func: Callable
    f_args: list[str] = field(default_factory=list)
    only_valid: bool = True

    def __str__(self):
        return f"{self.name} ({self.units})"

    def __call__(self, *args):
        return self.update_func(*args)


class Eval:
    def __init__(self, conf, s_configs):
        self.conf = conf
        self.comp_param = s_configs[0].get("comparison_param")
        self.feats = {
            "interactions": Feature(get_interactions),
            "int_costs": Feature(get_int_costs, ["interactions"]),
            "goal_inferences": Feature(get_goal_inference, ["int_costs"]),
            "traj_inferences": Feature(get_traj_inference, ["interactions", "int_costs"]),
            "mpd": Feature(get_mpd, ["interactions"]),
        }
        self.metrics = {
            "extra_ttg": Metric("Extra Time-to-Goal", "%", 2, min, eval_extra_ttg),
            "others_extra_ttg": Metric("Others' Extra Time-to-Goal", "%", 2, min, eval_others_extra_ttg, only_valid=False),
            "extra_dist": Metric("Extra Distance", "m", 2, min, eval_extra_dist),
            "others_extra_dist": Metric("Others' Extra Distance", "m", 2, min, eval_others_extra_dist, only_valid=False),
            "failure": Metric("Failure Rate", "%", 0, min, eval_failure, only_valid=False),
            "others_failure": Metric("Others' Failure Rate", "%", 0, min, eval_others_failure, only_valid=False),
            "efficiency": Metric("Path Efficiency", "%", 2, max, eval_efficiency),
            "others_efficiency": Metric("Others' Path Efficiency", "%", 2, max, eval_others_efficiency),
            "irregularity": Metric("Path Irregularity", "rad/m", 4, min, eval_irregularity),
            "others_irregularity": Metric("Others' Path Irregularity", "rad/m", 4, min, eval_others_irregularity),
            "legibility": Metric(
                "Legibility", "%", 2, max, eval_legibility, ["interactions", "goal_inferences"]
            ),
            "predictability": Metric(
                "Predictability", "%", 2, max, eval_predictability, ["traj_inferences"]
            ),
            "min_pass_inf": Metric(
                "Minimum Passing Inference",
                "%",
                2,
                max,
                eval_min_pass_inf,
                ["interactions", "goal_inferences"],
            ),
            "avg_min_pass_inf": Metric(
                "Average Minimum Passing Inference",
                "%",
                2,
                max,
                eval_avg_min_pass_inf,
                ["interactions", "goal_inferences"],
            ),
        }
        self.tracked_metrics = {
            "extra_ttg": Metric("Extra Time-to-Goal", "%", 2, min, eval_tracked_extra_ttg),
            "extra_dist": Metric("Extra Distance", "%", 2, min, eval_tracked_extra_dist),
            "mpd": Metric("Minmal Predicted Distance", "m", 2, min, eval_mpd, ["mpd"]),
            "pass_inf": Metric("Passing Inference", "%", 2, max, eval_pass_inf, ["interactions", "goal_inferences"]),
        }
        base_df = pd.DataFrame(s_configs)
        for col in base_df:
            if isinstance(base_df[col][0], list):
                base_df[col] = base_df[col].apply(','.join)
        cols = ["i", "name", "policy"]
        self.base_df_cols = cols if self.comp_param is None else cols + [self.comp_param]
        base_df = base_df[[c for c in base_df if len(set(base_df[c])) > 1 or c in self.base_df_cols]]
        self.df = base_df.reindex(columns=base_df.columns.tolist() + list(self.metrics))
        self.tracked_dfs = {}
        self.logger = logging.getLogger(__name__)

    def evaluate(self, i, env, fname):
        for f_v in self.feats.values():
            f_v.val = f_v(self.conf, env, *[self.feats[k].val for k in f_v.f_args])
        for m_k, m_v in self.metrics.items():
            self.df.at[i, m_k] = m_v(self.conf, env, *[self.feats[k].val for k in m_v.f_args])
        index = [(a.id, t) for a in env.agents.values() for t in np.around(np.linspace(0, env.max_duration, env.max_step + 1), 3)]
        index = pd.MultiIndex.from_tuples(index, names=("agent_id", "time"))
        self.tracked_dfs[i] = pd.DataFrame(index=index, columns=list(self.tracked_metrics), dtype="float64")
        for t_k, t_v in self.tracked_metrics.items():
            for a_id, a_vals in t_v(self.conf, env, *[self.feats[k] for k in t_v.f_args]).items():
                self.tracked_dfs[i].loc[zip(np.full(a_vals[0].shape, a_id), np.around(a_vals[0], 3)), t_k] = a_vals[1]
        for col in self.base_df_cols:
            self.tracked_dfs[i][col] = self.df.loc[i, col]

    def print_df(self, fname, df):
        means = df.groupby("policy", as_index=False).mean()
        for k, v in [(k, v) for k, v in self.metrics.items() if k in means]:
            means[k] = means[k].round(v.decimals)
            means[k] = means[k].map(lambda x: f"<{x}>" if x == v.opt_func(means[k]) else x)
        colalign = ["left" if k == "policy" else "right" for k in means]
        md = means.to_markdown(index=False, colalign=colalign)
        if self.conf["eval"]["show_tbl"]:
            print(fname, md, sep="\n")
        if self.conf["eval"]["save_tbl"]:
            os.makedirs(self.conf["eval"]["tbl_dir"], exist_ok=True)
            means = means.applymap(lambda x: str(x).replace("<", "\\textbf{").replace(">", "}"))
            buf = os.path.join(self.conf["eval"]["tbl_dir"], f"{fname}.latex")
            means.style.to_latex(buf=buf, hrules=True)

    def plot_df(self, fname, df):
        palette = {p: self.conf["agent"][p]["color"] for p in list(set(self.df["policy"]))}
        for k, v in [(k, v) for k, v in self.metrics.items() if k in df]:
            ax = sns.barplot(x=self.comp_param, y=k, hue="policy", data=df, palette=palette)
            for text in ax.legend_.texts:
                text.set_text(text._text.replace("_", " ").title())
            # if k != "min_pass_inf":
            #     ax.legend_.remove()
            ax.set_title(v.name)
            ax.set_ylabel(f"{v.name} ({v.units.replace('%', '%%')})")
            ax.set_ylabel("{name} ({units})".format(name=v.name, units=v.units.replace("%", "\\%")))
            ax.set_xlabel(self.comp_param.replace("_", " ").title())
            if self.conf["eval"]["save_bar_chart"]:
                os.makedirs(self.conf["eval"]["bar_chart_dir"], exist_ok=True)
                buf = os.path.join(self.conf["eval"]["bar_chart_dir"], f"{fname}_{k}_bar_chart.pdf")
                plt.savefig(buf)
            if self.conf["eval"]["show_bar_chart"]:
                plt.show()
            else:
                plt.close()

    def save_df(self, fname):
        os.makedirs(self.conf["eval"]["df_dir"], exist_ok=True)
        buf = os.path.join(self.conf["eval"]["df_dir"], f"{fname}_df.csv")
        if self.conf["eval"]["save_df"]:
            self.df.to_csv(buf)

    def plot_tracked_metrics(self, fname, df):
        palette = {p: self.conf["agent"][p]["color"] for p in list(set(df["policy"]))}
        iters = df["i"].drop_duplicates().to_list()
        policies = df["policy"].drop_duplicates().to_list()
        x = np.linspace(0, 1, len(df.time.drop_duplicates().index))
        for k, v in [(k, v) for k, v in self.tracked_metrics.items() if k in df]:
            for i, iter in enumerate(iters):
                for policy in policies:
                    idx = (df.policy == policy) & (df.i == iter)
                    for j, a_id in enumerate(df.loc[idx].agent_id.drop_duplicates().to_list()):
                        legend = False if i or j else "auto"
                        a_idx = idx & (df.agent_id == a_id)
                        a_idx_finite = a_idx & np.isfinite(df.loc[a_idx, k])
                        df.loc[a_idx_finite, "time"] -= df.loc[a_idx_finite, "time"].min()
                        df.loc[a_idx_finite, "time"] /= df.loc[a_idx_finite, "time"].max()
                        if a_idx_finite.any():
                            df.loc[a_idx, k] = np.interp(x, df.loc[a_idx_finite, "time"], df.loc[a_idx_finite, k])
                        df.loc[a_idx, "time"] = x
                        sns.lineplot(x="time", y=k, hue="policy", data=df[a_idx], palette=palette, estimator=None, legend=legend)
            plt.title(fname)
            plt.ylabel(str(v))
            if self.conf["eval"]["save_tracked_metrics"]:
                os.makedirs(self.conf["eval"]["tracked_metrics_dir"], exist_ok=True)
                buf = os.path.join(self.conf["eval"]["tracked_metrics_dir"], f"{fname}_{k}.pdf")
                plt.savefig(buf)
            if self.conf["eval"]["show_tracked_metrics"]:
                plt.show()
            else:
                plt.close()

            # Aggregate
            ax = sns.lineplot(x="time", y=k, hue="policy", data=df, palette=palette)
            ax.set_ylim(ymin=0)
            plt.title(f"agg_{fname}")
            plt.ylabel(str(v))
            if self.conf["eval"]["save_tracked_metrics"]:
                os.makedirs(self.conf["eval"]["tracked_metrics_dir"], exist_ok=True)
                buf = os.path.join(self.conf["eval"]["tracked_metrics_dir"], f"agg_{fname}_{k}.pdf")
                plt.savefig(buf)
            if self.conf["eval"]["show_tracked_metrics"]:
                plt.show()
            else:
                plt.close()

    def get_summary(self, s_name):
        df = self.df.copy()
        for k in [k for k, v in self.metrics.items() if k in df and v.units == '%']:
            df[k] *= 100
        if self.conf["eval"]["only_valid"]:
            invalid_idx = df[df.failure == 1].i.drop_duplicates().to_list()
            metrics = [m_k for m_k, m_v in self.metrics.items() if m_v.only_valid]
            df.loc[df.index[df.i.isin(invalid_idx)], metrics] = np.nan
        first_cols = ["policy", self.comp_param] if self.comp_param else ["policy"]
        df = df[first_cols + list(set(self.conf["eval"]["metrics"]) & set(df))]
        if self.conf["eval"]["show_tbl"] or self.conf["eval"]["save_tbl"]:
            if self.comp_param is None:
                self.print_df(s_name, df)
            else:
                for v in df[self.comp_param].drop_duplicates().to_list():
                    self.print_df(f"{s_name}_{v}", df[df[self.comp_param] == v])
        if self.conf["eval"]["show_bar_chart"] or self.conf["eval"]["save_bar_chart"]:
            self.plot_df(s_name, df)
        if self.conf["eval"]["save_df"]:
            self.save_df(s_name)
        if self.conf["eval"]["show_tracked_metrics"] or self.conf["eval"]["save_tracked_metrics"]:
            tracked_dfs = pd.concat(self.tracked_dfs.values()).reset_index()
            tracked_dfs = tracked_dfs[first_cols + ["agent_id", "time", "i"] + [m for m in self.conf["eval"]["tracked_metrics"] if m in tracked_dfs]]
            if self.comp_param is None:
                self.plot_tracked_metrics(s_name, tracked_dfs)
            else:
                for v in tracked_dfs[self.comp_param].drop_duplicates().to_list():
                    self.plot_tracked_metrics(f"{s_name}_{v}", tracked_dfs[tracked_dfs[self.comp_param] == v])
