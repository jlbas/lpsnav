import colorsys
import copy
import os

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Polygon
from utils import helper

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.labelsize": 5,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


def snapshot(ego_agent, id, agent):
    plt.close("all")
    plt.style.use("dark_background")
    _, ax = plt.subplots()
    ax.axis("equal")
    plt.ion()
    ax.scatter(*ego_agent.int_lines[id].T, color="gray")
    ax.scatter(*ego_agent.pred_int_lines[id].T, color="red")
    plt.pause(0.1)
    ax.scatter(
        *ego_agent.pred_int_lines[id].T,
        color=agent.color,
    )
    plt.pause(0.1)
    ax.scatter(*ego_agent.pos, color=ego_agent.color)
    plt.pause(0.1)
    ax.scatter(*ego_agent.goal, color=ego_agent.color)
    plt.pause(0.1)
    # for row in ego_agent.scaled_abs_prims[id]:
    for row, col_row in zip(ego_agent.abs_prims, ego_agent.col_mask):
        for pos, col in zip(row, col_row):
            if col:
                ax.scatter(*pos, c="red")
            else:
                ax.scatter(*pos, c="green")
            plt.pause(0.1)
    plt.pause(0.1)
    ax.scatter(*ego_agent.pos, c=ego_agent.color, s=100)
    plt.pause(0.1)
    ax.scatter(*agent.pos, c=agent.color, s=100)
    plt.pause(0.3)
    ax.add_patch(Circle(agent.pos, ego_agent.col_width[id], fill=False, ec="red"))
    plt.pause(0.3)


def snap(*agents, istep=0, fstep=-1, ion=False):
    if fstep == -1:
        fstep = agents[0].env.step + 1
    else:
        fstep += 1
    plt.style.use("dark_background")
    _, ax = plt.subplots()
    ax.axis("equal")
    if ion:
        plt.ion()
        for a in agents:
            ax.scatter(*a.goal, c=a.color)
            plt.pause(0.1)
            ax.plot(a.pos_log[istep:fstep, 0], a.pos_log[istep:fstep, 1], c=a.color, lw=3)
            plt.pause(0.1)
        plt.close("all")
    else:
        for a in agents:
            ax.scatter(*a.goal, c=a.color)
            ax.plot(a.pos_log[istep:fstep, 0], a.pos_log[istep:fstep, 1], c=a.color, lw=3)
        plt.show()


class Animate:
    def __init__(self, config, scenario, iter):
        self.config = config
        self.scenario = scenario
        self.iter = iter
        self.agents_log = {}

    def ani(self, i, agents, ego_agent, last_frame, eval, plt, fig, pdf):
        for a in agents:
            a.patches.path.set_xy(a.pos_log[: i + 1])
            a.patches.triangle.set_xy(helper.rotate(a.body_coords, a.heading_log[i]) + a.pos_log[i])
            a.patches.body.center = a.pos_log[i]
            if self.config.animation.debug:
                if a.policy == "lpnav" and a.is_ego:
                    for log, circles in zip(a.int_lines_log.values(), a.patches.int_lines):
                        for pos, c in zip(log[i], circles):
                            c.set_radius(0) if np.isinf(pos[0]) else c.set(center=pos, radius=0.05)
                    for log, circles in zip(
                        a.pred_int_lines_log.values(), a.patches.pred_int_lines
                    ):
                        for pos, c in zip(log[i], circles):
                            c.set_radius(0) if np.isinf(pos[0]) else c.set(center=pos, radius=0.05)
                    for log, c in zip(a.col_circle_log.values(), a.patches.col_circle):
                        c.set(alpha=0) if np.isinf(log[i][0]) else c.set(center=log[i], alpha=1)
                    for j, (prim, speed, col_row) in enumerate(
                        zip(a.patches.prims, a.abs_prims_log[i], a.col_mask_log[i])
                    ):
                        for k, (pt, coord, col) in enumerate(zip(prim, speed, col_row)):
                            pt.set_center(coord)
                            pt.set_color("red" if col else "green")
                            is_opt = [j, k] == a.opt_log[i]
                            pt.set_zorder(1 if is_opt else 0)
                            pt.set_radius(0.08 if is_opt else 0.04)
                if a.policy == "sfm":
                    a.patches.f_goal.set_xy([a.pos_log[i], a.pos_log[i] + a.goal_force_log[i]])
                    a.patches.f_ped.set_xy([a.pos_log[i], a.pos_log[i] + a.ped_force_log[i]])
                    a.patches.f_tot.set_xy([a.pos_log[i], a.pos_log[i] + a.tot_force_log[i]])
        pdf is not None and pdf.savefig(fig)
        i == last_frame - 1 and self.config.animation.autoplay and plt.close(fig)
        return helper.flatten([p for a in agents for p in a.patches])

    def init_ani(self, agents, ego_agent=None, eval=None, fname=None):
        self.config.animation.dark_bg and plt.style.use("dark_background")
        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        fig.set_size_inches(9, 9)
        ax.axis("scaled")
        ax.set(title=self.iter)
        ax.axis("off")
        x, y = np.concatenate([a.pos_log for a in agents]).T
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        pad = 2 * self.config.agent.radius
        ax.axis([x_min - pad, x_max + pad, y_min - pad, y_max + pad])
        for i, a in enumerate(agents):
            a.patches.goal = Circle((a.goal), 0.05, color=a.color, fill=False, lw=3, zorder=1)
            a.patches.path = Polygon(
                ((0, 0), (0, 0)),
                closed=False,
                fill=False,
                lw=5,
                zorder=0,
                color=a.color,
                capstyle="round",
            )
            r = 0.9 * a.radius
            x = r * np.cos(np.pi / 6)
            y = r * np.sin(np.pi / 6)
            a.body_coords = [(r, 0), (-x, y), (-x, -y)]
            a.patches.triangle = Polygon(a.body_coords, fc=ax.get_facecolor(), lw=4, zorder=i + 3)
            a.patches.body = Circle((a.pos), a.radius, color=a.color, zorder=i + 2)
            if self.config.animation.debug:
                if a.policy == "lpnav" and a.is_ego:
                    c = (0, 0)
                    a.patches.prims = [[Circle(c) for _ in a.rel_headings] for _ in a.speeds]
                    a.patches.int_lines = [[Circle(c), Circle(c)] for _ in a.int_lines_log]
                    a.patches.pred_int_lines = copy.deepcopy(a.patches.int_lines)
                    cws = a.col_width.values()
                    a.patches.col_circle = [Circle(c, cw, fill=False, ec="red") for cw in cws]
                if a.policy == "sfm":
                    p = Polygon(((0, 0), (0, 0)), closed=False, fill=False, lw=3, zorder=10)
                    for attr, c in zip(["f_goal", "f_ped", "f_tot"], ["blue", "red", "purple"]):
                        poly = copy.deepcopy(p)
                        poly.set_color(c)
                        setattr(a.patches, attr, poly)
        for patch in helper.flatten([p for a in agents for p in a.patches]):
            ax.add_patch(patch)
        if fname is None:
            fname = f"{self.scenario}_overlay"
        fname = os.path.join(self.config.animation.ani_dir, fname)
        pdf = PdfPages(f"{fname}.pdf") if self.config.animation.save_ani_as_pdf else None
        frames = max([len(a.pos_log) for a in agents])
        ani = FuncAnimation(
            fig,
            self.ani,
            frames=frames,
            interval=int(1000 / self.config.animation.speed * self.config.env.dt),
            fargs=(agents, ego_agent, frames, eval, plt, fig, pdf),
            blit=True,
            repeat=False,
        )
        if self.config.animation.save_ani:
            os.makedirs(self.config.animation.ani_dir, exist_ok=True)
            fps = int(self.config.animation.speed / self.config.env.dt)
            ani.save(f"{fname}.mp4", writer="ffmpeg", fps=fps)
            plt.savefig(f"{fname}.pdf")
        self.config.animation.show_ani and plt.show()
        pdf is not None and pdf.close()

    def plot(self, agents, interaction=None, fname=None):
        self.config.animation.dark_bg and plt.style.use("dark_background")
        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        fig.set_size_inches(1.6, 1.6)
        ax.axis("square")
        # fig.subplots_adjust(left=0.15, right=0.97, bottom=0.06, top=1)
        ax.set(xlabel=r"$x$ (m)", ylabel=r"$y$ (m)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(length=0, pad=2)
        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1
        x, y = np.concatenate([a.pos_log for a in agents.values()]).T
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        pad = 2 * self.config.agent.radius
        ax.axis([x_min - pad, x_max + pad, y_min - pad, y_max + pad])
        ax.axis([-3.5 - pad, 3.5 + pad, -3.5 - pad, 3.5 + pad])
        step = int(self.config.animation.body_interval / self.config.env.dt)
        sample_slice = slice(None, None, step)
        first_inattentive = True
        for id, a in agents.items():
            ax.add_patch(
                Circle(
                    a.goal,
                    self.config.agent.goal_tol,
                    ec=a.color,
                    fill=None,
                    lw=0.5,
                    zorder=2 * len(agents) + id,
                )
            )
            if self.config.animation.plot_traj:
                if (a.policy != "inattentive") or (a.policy == "inattentive" and first_inattentive):
                    label = a.policy
                    first_inattentive = False
                else:
                    label = None
                ax.plot(
                    np.array(a.pos_log)[:, 0],
                    np.array(a.pos_log)[:, 1],
                    c=a.color,
                    lw=0.5,
                    solid_capstyle="round",
                    zorder=len(agents) + id,
                    label=label,
                )
            if self.config.animation.plot_body:
                hls_color = colorsys.rgb_to_hls(*mc.to_rgb(a.color))
                sampled_traj = a.pos_log[sample_slice]
                lightness_range = np.linspace(
                    hls_color[1] + 0.1 * (1 - hls_color[1]),
                    1 - 0.2 * (1 - hls_color[1]),
                    len(sampled_traj),
                )
                zorder = id if a.policy != "inattentive" else 0
                for j, (pos, lightness) in enumerate(zip(sampled_traj, lightness_range[::-1])):
                    if interaction:
                        start, end = interaction.int_idx.get(id, (-np.inf, np.inf) if a.is_ego else (0, 0))
                        s, ec = (hls_color[2], a.color) if start <= j * step <= end else (0, None)
                    else:
                        s, ec = (hls_color[2], a.color)
                    c = colorsys.hls_to_rgb(hls_color[0], lightness, s)
                    ax.add_patch(Circle(pos, a.radius, fc=c, ec=ec, lw=0.1, zorder=zorder))
        # ax.legend()
        if self.config.animation.save_plot:
            os.makedirs(self.config.animation.plot_dir, exist_ok=True)
            if fname is None:
                fname = f"{self.scenario}_overlay"
            fname = os.path.join(self.config.animation.plot_dir, fname)
            plt.savefig(fname + ".pdf")
        self.config.animation.show_plot and plt.show()

    def plot_inferences(self, policy, interaction, goal_inference, traj_inference, fname=None):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.set(
            title=f"{policy} Interaction Goal Inference",
            xlabel=r"Time (s)",
            ylabel=r"$P(\mathcal{I}_i\mid\xi_{s\rightarrow{t}})$",
        )
        ax2.set(
            title=f"{policy} Trajectory Inference",
            xlabel=r"Time (s)",
            ylabel=r"$P(\xi_{s\rightarrow{t}}\mid\mathcal{I}_i)$",
        )
        inf_iters = (interaction.agents, goal_inference.values(), traj_inference.values())
        for id, goal_scores, traj_scores in zip(*inf_iters):
            c = "w" if self.config.animation.dark_bg else "k"
            c = interaction.agents[id].color
            for infs, ax in zip((goal_scores, traj_scores), (ax1, ax2)):
                start, stop = np.multiply(self.config.env.dt, interaction.int_idx[id])
                num = np.diff(interaction.int_idx[id])[0] + 1
                t = np.linspace(start, stop, num, endpoint=True)
                step = int(self.config.animation.body_interval / self.config.env.dt)
                sample_slice = slice(None, None, step)
                sampled_t = t[sample_slice]
                passing_idx = 0 if interaction.passing_idx[id] == 0 else 2
                inf_idxs = [0, 1, 2, passing_idx]
                labels = ["left", "collide", "right", "pass"]
                lss = ["--", ":", "-"]
                lss.append(lss[passing_idx])
                for inf_idx, label, ls in zip(inf_idxs, labels, lss):
                    if label in self.config.animation.inferences:
                        label = f"{id} {label.replace('pass', labels[passing_idx])}"
                        ax.plot(t, infs[inf_idx], lw=2, color=c, label=label, ls=ls)
                        ax.legend()
                        sampled_inf = infs[inf_idx][sample_slice]
                        ax.scatter(sampled_t, sampled_inf, color=c, linewidths=2)
        if self.config.animation.save_inferences:
            os.makedirs(self.config.animation.inferences_dir, exist_ok=True)
            if fname is None:
                fname = f"{self.scenario}"
            fname = os.path.join(self.config.animation.inferences_dir, fname)
            for fig, suf in zip([fig1, fig2], ["goal_inf", "traj_inf"]):
                fig.savefig(f"{fname}_{suf}.pdf")
        self.config.animation.show_inferences and plt.show()

    def overlay(self):
        if self.config.animation.show_ani or self.config.animation.save_ani:
            self.init_ani(self.agents_log.values())
        if self.config.animation.show_plot or self.config.animation.save_plot:
            self.plot(self.agents_log)

    def animate(self, iter, agents, ego_agent, scenario, n, fname, eval=None):
        if self.config.animation.show_ani or self.config.animation.save_ani:
            self.init_ani(agents.values(), ego_agent, eval, fname)
        if self.config.animation.show_plot or self.config.animation.save_plot:
            self.plot(agents, eval.feats["interaction"].log[scenario][n][ego_agent.policy][iter], fname)
        if eval and (
            self.config.animation.show_inferences or self.config.animation.save_inferences
        ):
            self.plot_inferences(
                eval.policy_dict[ego_agent.policy],
                eval.feats["interaction"].log[scenario][n][ego_agent.policy][iter],
                eval.feats["goal_inference"].log[scenario][n][ego_agent.policy][iter],
                eval.feats["traj_inference"].log[scenario][n][ego_agent.policy][iter],
                fname,
            )
        self.agents_log.update(agents)
