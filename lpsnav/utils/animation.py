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
    def __init__(self, config):
        self.conf = config
        self.agents = {}
        self.agent_logs = {}
        self.conf["dark_background"] and plt.style.use("dark_background")
        plt.style.use("./config/paper.mplstyle")

    def ani(self, i, agents, logs, patches, last_frame, plt, fig):
        for (id, p), log in zip(patches.items(), logs.values()):
            p.path.set_xy(log.pos[: i + 1])
            p.triangle.set_xy(helper.rotate(agents[id].body_coords, log.heading[i]) + log.pos[i])
            p.body.center = log.pos[i]
        i == last_frame - 1 and self.conf["autoplay"] and plt.close(fig)
        return flatten([sub_p for p in patches.values() for sub_p in p])

    def init_ani(self, dt, agents, logs, fname):
        figsize = (1920 / self.conf["dpi"], 1080 / self.conf["dpi"])
        fig, ax = plt.subplots(constrained_layout=True, figsize=figsize, dpi=self.conf["dpi"])
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
        ax.axis("scaled")
        ax.axis("off")
        x, y = np.concatenate([log.pos for log in logs.values()]).T
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        pad = 4 * max([a.radius for a in agents.values()])
        ax.axis([x_min - pad, x_max + pad, y_min - pad, y_max + pad])
        patches = {id: Patches() for id in agents}
        for id, a in agents.items():
            patches[id].goal = Circle((a.goal), 0.05, color=a.color, fill=False, lw=3, zorder=1)
            patches[id].path = Polygon(
                ((0, 0), (0, 0)),
                closed=False,
                fill=False,
                lw=7,
                zorder=0,
                color=a.color,
                capstyle="round",
            )
            patches[id].triangle = Polygon(
                a.body_coords, fc=ax.get_facecolor(), lw=4, zorder=id + 3
            )
            patches[id].body = Circle((0, 0), a.radius, color=a.color, zorder=id + 2)
        for patch in flatten([p for id in agents for p in patches[id]]):
            ax.add_patch(patch)
        buf = os.path.join(self.conf["ani_dir"], fname)
        frames = max([len(log.pos) for log in logs.values()])
        ani = FuncAnimation(
            fig,
            self.ani,
            frames=frames,
            interval=int(1000 / self.conf["speed"] * dt),
            fargs=(agents, logs, patches, frames, plt, fig),
            blit=True,
            repeat=False,
        )
        if self.conf["save_ani"]:
            os.makedirs(self.conf["ani_dir"], exist_ok=True)
            fps = int(self.conf["speed"] / dt)
            ani.save(f"{buf}.mp4", writer="ffmpeg", fps=fps)
        if self.conf["save_ani_as_pdf"]:
            os.makedirs(self.conf["ani_dir"], exist_ok=True)
            ani.save(f"{buf}.pdf", writer="imagemagick")
        self.conf["show_ani"] and plt.show() or plt.close()

    def plot(self, dt, agents, logs, fname):
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
        x, y = np.concatenate([log.pos for log in logs.values()]).T
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        pad = 2 * max([a.radius for a in agents.values()])
        ax.axis([x_min - pad, x_max + pad, y_min - pad, y_max + pad])
        step = int(self.conf["body_interval"] / dt)
        sample_slice = slice(None, None, step)
        first_inattentive = True
        for (id, a), log in zip(agents.items(), logs.values()):
            ax.add_patch(
                Circle(
                    a.goal,
                    a.goal_tol,
                    ec=a.color,
                    fill=None,
                    lw=0.5,
                    zorder=2 * len(agents) + id,
                )
            )
            if self.conf["plot_traj"]:
                if (a.policy != "inattentive") or (a.policy == "inattentive" and first_inattentive):
                    label = a.policy
                    first_inattentive = False
                else:
                    label = None
                ax.plot(
                    np.array(log.pos)[:, 0],
                    np.array(log.pos)[:, 1],
                    c=a.color,
                    lw=0.5,
                    solid_capstyle="round",
                    zorder=len(agents) + id,
                    label=label,
                )
            if self.conf["plot_body"]:
                hls_color = colorsys.rgb_to_hls(*mc.to_rgb(a.color))
                sampled_traj = log.pos[sample_slice]
                lightness_range = np.linspace(
                    hls_color[1] + 0.1 * (1 - hls_color[1]),
                    1 - 0.2 * (1 - hls_color[1]),
                    len(sampled_traj),
                )
                zorder = id if a.policy != "inattentive" else 0
                for pos, lightness in zip(sampled_traj, lightness_range[::-1]):
                    s, ec = (hls_color[2], a.color)
                    c = colorsys.hls_to_rgb(hls_color[0], lightness, s)
                    ax.add_patch(Circle(pos, a.radius, fc=c, ec=ec, lw=0.1, zorder=zorder))
        if self.conf["save_plot"]:
            os.makedirs(self.conf["plot_dir"], exist_ok=True)
            plt.savefig(os.path.join(self.conf["plot_dir"], f"{fname}.pdf"))
        self.conf["show_plot"] and plt.show() or plt.close()

    def overlay(self, dt, fname):
        max_len = max([len(getattr(log, k)) for log in self.agent_logs.values() for k in vars(log)])
        for log in self.agent_logs.values():
            for k in vars(log):
                arr = getattr(log, k)
                pad_len = max_len - len(arr)
                pad_width = (0, pad_len) if arr.ndim == 1 else [(0, pad_len), (0, 0)]
                setattr(log, k, np.pad(arr, pad_width, mode="edge"))
        self.animate(dt, self.agents, self.agent_logs, fname, overlay=True)
        self.agents.clear()
        self.agent_logs.clear()

    def animate(self, dt, agents, logs, fname, overlay):
        if any((self.conf["show_ani"], self.conf["save_ani"], self.conf["save_ani_as_pdf"])):
            self.init_ani(dt, agents, logs, fname)
        if any((self.conf["show_plot"], self.conf["save_plot"])):
            self.plot(dt, agents, logs, fname)
        if overlay:
            self.agents.update(agents)
            self.agent_logs.update(logs)
