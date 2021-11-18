import colorsys
import os

import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle, Polygon
from utils import helper


def snapshot(ego_agent, id):
    if ego_agent.config.dark_bkg:
        plt.style.use("dark_background")
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    # fig.canvas.manager.window.wm_geometry("+1150+950")
    fig.canvas.manager.window.wm_geometry("+1510+450")
    ax.axis("equal")
    plt.ion()
    ax.scatter(ego_agent.int_lines[id][:, 0], ego_agent.int_lines[id][:, 1], color="gray")
    ax.scatter(ego_agent.pred_int_lines[id][:, 0], ego_agent.pred_int_lines[id][:, 1], color="red")
    plt.pause(0.1)
    ax.scatter(
        ego_agent.pred_int_lines[id][:, 0],
        ego_agent.pred_int_lines[id][:, 1],
        color=ego_agent.other_agents[id].color,
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
    plt.close("all")


def snap(*agents, istep=0, fstep=-1):
    if agents[0].config.dark_bkg:
        plt.style.use("dark_background")
    _, ax = plt.subplots()
    ax.axis("equal")
    if fstep == -1:
        fstep = agents[0].env.step + 1
    else:
        fstep += 1
    for a in agents:
        ax.scatter(*a.goal, c=a.color)
        ax.plot(a.pos_log[istep:fstep, 0], a.pos_log[istep:fstep, 1], c=a.color, lw=3)
    plt.show()


class Animate:
    def __init__(self, config, scenario):
        self.config = config
        self.scenario = scenario
        self.agents_log = dict()

    def ani(self, i, agents, last_frame, plt, fig):
        for a in agents:
            a.patches.path.set_xy(a.pos_log[: i + 1])
            try:
                a.patches.body_poly.set_xy(
                    helper.rotate(a.body_coords, a.heading_log[i]) + a.pos_log[i]
                )
                a.patches.body.center = a.pos_log[i]
            except IndexError:
                pass

            if self.config.debug:
                if a.policy == "legitable" and i < min(
                    [len(log) - 1 for log in a.int_lines_log.values()]
                ):
                    for log, circles in zip(a.int_lines_log.values(), a.patches.int_lines):
                        for pos, circle in zip(log[i], circles):
                            if pos is None or np.all(np.isnan(pos)):
                                circle.set_radius(0)
                            else:
                                circle.set_radius(0.05)
                                circle.center = pos
                    for log, circles in zip(
                        a.pred_int_lines_log.values(), a.patches.pred_int_lines
                    ):
                        for pos, circle in zip(log[i], circles):
                            if pos is None or np.all(np.isnan(pos)):
                                circle.set_radius(0)
                            else:
                                circle.set_radius(0.05)
                                circle.center = pos
                if a.policy == "legitable":
                    for j, (prim, speed, col_row) in enumerate(
                        zip(a.patches.prims, a.abs_prims_log[i], a.col_mask_log[i])
                    ):
                        for k, (pt, coord, col) in enumerate(zip(prim, speed, col_row)):
                            pt.center = coord
                            zorder, r = [1, 0.08] if [j, k] == a.opt_log[i] else [0, 0.04]
                            pt.set_zorder(zorder)
                            pt.set_radius(r)
                            fc = "lightgray" if [j, k] == a.opt_log[i] else "#004D40"
                            fc = "red" if col else "green"
                            pt.set_facecolor(fc)

        if i == last_frame - 1:
            pass
            # plt.close(fig)

        return helper.flatten([p for a in agents for p in a.patches])

    def init_ani(self, agents, filename=None):
        if self.config.dark_bkg:
            plt.style.use("dark_background")
        fig, ax = plt.subplots()
        fig.tight_layout()
        fig.set_size_inches(16, 9)
        # fig.subplots_adjust(left=0.05, right=1, bottom=0.075, top=0.95, wspace=0, hspace=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("scaled")
        # ax.axis('off')
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        pos_logs = np.concatenate([a.pos_log for a in agents])
        x_min = np.min(pos_logs[:, 0])
        x_max = np.max(pos_logs[:, 0])
        y_min = np.min(pos_logs[:, 1])
        y_max = np.max(pos_logs[:, 1])
        ax.axis([x_min - 2, x_max + 2, y_min - 2, y_max + 2])
        # colors = sns.color_palette(n_colors=len(agents))
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
            x = 0.9 * a.radius * np.cos(np.pi / 6)
            y = 0.9 * a.radius * np.sin(np.pi / 6)
            a.body_coords = [(0.9 * a.radius, 0), (-x, y), (-x, -y)]
            a.patches.body_poly = Polygon(
                a.body_coords, facecolor=ax.get_facecolor(), linewidth=4, zorder=i + 3
            )
            a.patches.body = Circle(
                (a.pos), a.radius, color=a.color, zorder=i + 2, label=f"{type(a).__name__}"
            )
            if self.config.debug:
                if a.policy == "legitable":
                    a.patches.prims = [
                        [Circle((0, 0), 0.04, color="#004D40", lw=0) for _ in a.rel_headings]
                        for _ in a.speeds
                    ]
                if a.policy == "legitable":
                    a.patches.int_lines = [
                        [
                            Circle((0, 0), 0, color="gray"),
                            Circle((0, 0), 0, color="gray"),
                        ]
                        for _ in a.int_lines_log
                    ]
                    a.patches.pred_int_lines = [
                        [
                            Circle((0, 0), 0, color=a.other_agents[id].color)
                            for _ in a.pred_int_lines_log[id][0]
                        ]
                        for id in a.pred_int_lines_log
                    ]
        for patch in helper.flatten([p for a in agents for p in a.patches]):
            ax.add_patch(patch)
        ax.legend()

        ani = FuncAnimation(
            fig,
            self.ani,
            frames=max([len(a.pos_log) for a in agents]),
            interval=int(1000 * self.config.timestep),
            fargs=(agents, max([len(a.pos_log) for a in agents]), plt, fig),
            blit=True,
            repeat=False,
        )
        if self.config.show_ani:
            plt.show()

        if self.config.save_ani:
            os.makedirs(self.config.ani_dir, exist_ok=True)
            if filename is None:
                filename = f"{self.scenario}_overlay"
            vidname = os.path.join(self.config.ani_dir, f"{filename}.mp4")
            ani.save(vidname, writer=FFMpegWriter(fps=int(1 / self.config.timestep)))

    def plot(self, agents, filename=None):
        if self.config.dark_bkg:
            plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times"],
                "xtick.labelsize": 5,
                "ytick.labelsize": 5,
                "axes.labelsize": 5,
            }
        )
        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        fig.set_size_inches(1.6, 1.6)
        ax.axis("square")
        # fig.subplots_adjust(left=0.15, right=0.97, bottom=0.06, top=1)
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$y$ (m)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(length=0, pad=2)
        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1
        pos_logs = np.concatenate([a.pos_log for a in agents])
        x_min = np.min(pos_logs[:, 0])
        x_max = np.max(pos_logs[:, 0])
        y_min = np.min(pos_logs[:, 1])
        y_max = np.max(pos_logs[:, 1])
        padding = 2 * self.config.radius
        ax.axis([x_min - padding, x_max + padding, y_min - padding, y_max + padding])
        sampled_len = 0
        for a in agents:
            a.sampled_traj = a.pos_log[:: int(self.config.body_interval / self.config.timestep)]
            sampled_len = max(sampled_len, len(a.sampled_traj))
        first_inattentive = True
        for i, a in enumerate(list(agents)[::-1]):
            ax.add_patch(
                Circle(
                    a.goal,
                    self.config.goal_tol,
                    ec=a.color,
                    fill=None,
                    lw=0.5,
                    zorder=2 * len(agents) + i,
                )
            )
            if self.config.plot_traj:
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
                    zorder=len(agents) + i,
                    label=label,
                )
            if self.config.plot_body:
                hls_color = colorsys.rgb_to_hls(*mc.to_rgb(a.color))
                lightness_range = np.linspace(
                    hls_color[1] + 0.1 * (1 - hls_color[1]),
                    1 - 0.2 * (1 - hls_color[1]),
                    sampled_len,
                )
                zorder = i if a.policy != "inattentive" else 0
                for pos, lightness in zip(a.sampled_traj, lightness_range[::-1]):
                    c = colorsys.hls_to_rgb(hls_color[0], lightness, hls_color[2])
                    ax.add_patch(
                        Circle(pos, self.config.radius, fc=c, ec=a.color, lw=0.1, zorder=zorder)
                    )
        # ax.legend()
        if self.config.save_plot:
            os.makedirs(self.config.plot_dir, exist_ok=True)
            if filename is None:
                filename = f"{self.scenario}_overlay"
            plotname = os.path.join(self.config.plot_dir, filename)
            plt.savefig(plotname + ".pdf", backend="pgf")
        if self.config.show_plot:
            plt.show()

    def overlay(self):
        if self.config.show_ani or self.config.save_ani:
            self.init_ani(self.agents_log.values())
        if self.config.show_plot or self.config.save_plot:
            self.plot(self.agents_log.values())

    def animate(self, env):
        if self.config.show_ani or self.config.save_ani:
            self.init_ani(env.agents.values(), str(env))
        if self.config.show_plot or self.config.save_plot:
            self.plot(env.agents.values(), str(env))
        self.agents_log.update(env.agents)
