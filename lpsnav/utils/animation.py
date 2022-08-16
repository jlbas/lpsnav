import colorsys
from dataclasses import dataclass
import os

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from utils import helper
from utils.utils import flatten


@dataclass
class Patches:
    def __iter__(self):
        for _, val in self.__dict__.items():
            yield val


class Animate:
    def __init__(self, conf):
        self.show_ani = conf["show_ani"]
        self.save_ani = conf["save_ani"]
        self.save_ani_as_pdf = conf["save_ani_as_pdf"]
        self.ani_dir = conf["ani_dir"]
        self.show_plot = conf["show_plot"]
        self.save_plot = conf["save_plot"]
        self.plot_dir = conf["plot_dir"]
        self.plot_body = conf["plot_body"]
        self.body_interval = conf["body_interval"]
        self.plot_traj = conf["plot_traj"]
        self.speed = conf["speed"]
        self.autoplay = conf["autoplay"]
        self.dpi = conf["dpi"]
        self.follow_ego = conf["follow_ego"]
        self.agents = {}
        self.agent_logs = {}
        conf["dark_background"] and plt.style.use("dark_background")
        plt.style.use("./config/paper.mplstyle")

    def ani(self, i, agents, ego_id, logs, patches, walls, wall_plots, last_frame, plt, fig):
        for (k, p), log in zip(patches.items(), logs.values()):
            if self.follow_ego and ego_id is not None:
                if k == ego_id:
                    p.triangle.set_xy(helper.rotate(agents[k].body_coords, log.heading[i]))
                    p.path.set_xy(log.pos[: i + 1] - log.pos[i])
                else:
                    rel_pos = log.pos[i] - logs[ego_id].pos[i]
                    p.triangle.set_xy(helper.rotate(agents[k].body_coords, log.heading[i]) + rel_pos)
                    p.body.center = rel_pos
                    p.path.set_xy(log.pos[: i + 1] - logs[ego_id].pos[i])
                p.goal.center = agents[k].goal - logs[ego_id].pos[i]
                for wall, wall_plt in zip(walls, wall_plots):
                    wall_plt.set_data(*np.transpose(wall - logs[ego_id].pos[i]))
            else:
                p.triangle.set_xy(helper.rotate(agents[k].body_coords, log.heading[i]) + log.pos[i])
                p.body.center = log.pos[i]
                p.path.set_xy(log.pos[: i + 1])
        i == last_frame - 1 and self.autoplay and plt.close(fig)
        return flatten([sub_p for p in patches.values() for sub_p in p] + wall_plots)

    def init_ani(self, dt, ego_id, agents, logs, walls, fname):
        figsize = (1920 / self.dpi, 1080 / self.dpi)
        fig, ax = plt.subplots(constrained_layout=True, figsize=figsize, dpi=self.dpi)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        ax.axis("scaled")
        ax.axis("off")
        if self.follow_ego:
            x = 5
            ratio = figsize[0] / figsize[1]
            ax.axis([-x, x, -x / ratio, x / ratio])
        else:
            a_pos = [log.pos for log in logs.values()]
            wall_pos = list(np.reshape(walls, (1, -1, 2)))
            x, y = np.concatenate(a_pos + wall_pos).T
            x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
            pad = 4 * max([a.radius for a in agents.values()])
            ax.axis([x_min - pad, x_max + pad, y_min - pad, y_max + pad])
        patches = {k: Patches() for k in agents}
        for k, a in agents.items():
            patches[k].goal = Circle((a.goal), 0.05, color=a.color, fill=False, lw=3, zorder=1)
            patches[k].path = Polygon(
                ((0, 0), (0, 0)),
                closed=False,
                fill=False,
                lw=7,
                zorder=0,
                color=a.color,
                capstyle="round",
            )
            patches[k].triangle = Polygon(
                a.body_coords, fc=ax.get_facecolor(), lw=4, zorder=k + 3
            )
            patches[k].body = Circle((0, 0), a.radius, color=a.color, zorder=k + 2)
        for patch in flatten([p for k in agents for p in patches[k]]):
            ax.add_patch(patch)
        wall_plots = [ax.plot(*np.transpose(wall), lw=5, c="sienna", solid_capstyle="round")[0] for wall in walls]
        buf = os.path.join(self.ani_dir, fname)
        frames = max([len(log.pos) for log in logs.values()])
        ani = FuncAnimation(
            fig,
            self.ani,
            frames=frames,
            interval=int(1000 / self.speed * dt),
            fargs=(agents, ego_id, logs, patches, walls, wall_plots, frames, plt, fig),
            blit=True,
            repeat=False,
        )
        if self.save_ani:
            os.makedirs(self.ani_dir, exist_ok=True)
            fps = int(self.speed / dt)
            ani.save(f"{buf}.mp4", writer="ffmpeg", fps=fps)
        if self.save_ani_as_pdf:
            os.makedirs(self.ani_dir, exist_ok=True)
            ani.save(f"{buf}.pdf", writer="imagemagick")
        self.show_ani and plt.show() or plt.close()

    def plot(self, dt, agents, logs, walls, fname):
        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        ax.axis("square")
        ax.set(xlabel=r"$x$ (m)", ylabel=r"$y$ (m)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(length=0, pad=2)
        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1
        a_pos = [log.pos for log in logs.values()]
        wall_pos = list(np.reshape(walls, (1, -1, 2)))
        x, y = np.concatenate(a_pos + wall_pos).T
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        pad = 2 * max([a.radius for a in agents.values()])
        ax.axis([x_min - pad, x_max + pad, y_min - pad, y_max + pad])
        step = int(self.body_interval / dt)
        sample_slice = slice(None, None, step)
        first_inattentive = True
        for (k, a), log in zip(agents.items(), logs.values()):
            ax.add_patch(
                Circle(
                    a.goal,
                    a.goal_tol,
                    ec=a.color,
                    fill=None,
                    lw=1,
                    zorder=2 * len(agents) + k,
                )
            )
            if self.plot_traj:
                if (a.policy != "inattentive") or (a.policy == "inattentive" and first_inattentive):
                    label = a.policy
                    first_inattentive = False
                else:
                    label = None
                ax.plot(
                    np.array(log.pos)[:, 0],
                    np.array(log.pos)[:, 1],
                    c=a.color,
                    lw=1,
                    solid_capstyle="round",
                    zorder=len(agents) + k,
                    label=label,
                )
            if self.plot_body:
                hls_color = colorsys.rgb_to_hls(*mc.to_rgb(a.color))
                sampled_traj = log.pos[sample_slice]
                lightness_range = np.linspace(
                    hls_color[1] + 0.1 * (1 - hls_color[1]),
                    1 - 0.2 * (1 - hls_color[1]),
                    len(sampled_traj),
                )
                zorder = k if a.policy != "inattentive" else -1
                for pos, lightness in zip(sampled_traj, lightness_range[::-1]):
                    s, ec = (hls_color[2], a.color)
                    c = colorsys.hls_to_rgb(hls_color[0], lightness, s)
                    ax.add_patch(Circle(pos, a.radius, fc=c, ec=ec, lw=0.1, zorder=zorder))
        for wall in walls:
            ax.plot(*np.transpose(wall), lw=5, c="sienna", solid_capstyle="round")
        if self.save_plot:
            os.makedirs(self.plot_dir, exist_ok=True)
            plt.savefig(os.path.join(self.plot_dir, f"{fname}.pdf"))
        self.show_plot and plt.show() or plt.close()

    def overlay(self, dt, walls, fname):
        max_len = max([len(getattr(log, k)) for log in self.agent_logs.values() for k in vars(log)])
        for log in self.agent_logs.values():
            for k in vars(log):
                arr = getattr(log, k)
                pad_len = max_len - len(arr)
                pad_width = (0, pad_len) if arr.ndim == 1 else [(0, pad_len), (0, 0)]
                setattr(log, k, np.pad(arr, pad_width, mode="edge"))
        self.animate(dt, self.agents, self.agent_logs, walls, fname, overlay=True)
        self.agents.clear()
        self.agent_logs.clear()

    def animate(self, dt, agents, logs, walls, fname, overlay, ego_id=None):
        if any((self.show_ani, self.save_ani, self.save_ani_as_pdf)):
            self.init_ani(dt, ego_id, agents, logs, walls, fname)
        if any((self.show_plot, self.save_plot)):
            self.plot(dt, agents, logs, walls, fname)
        if overlay:
            self.agents.update(agents)
            self.agent_logs.update(logs)
