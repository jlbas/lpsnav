from dataclasses import dataclass

import numpy as np
from utils import helper


@dataclass
class Patches:
    def __iter__(self):
        for _, val in self.__dict__.items():
            yield val


class Agent:
    def __init__(self, config, env, id, policy, start, goal=None, max_speed=None):
        self.env_conf = config.env
        self.env = env
        self.id = id
        self.policy = policy
        self.start = np.array(start, dtype=float)
        self.goal = self.start if goal is None else np.array(goal, dtype=float)
        self.conf = getattr(config, self.policy)
        self.radius = self.conf.radius
        self.min_speed = self.conf.min_speed
        self.max_speed = self.conf.max_speed if max_speed is None else max_speed
        self.max_accel = self.conf.max_accel
        self.max_ang_accel = self.conf.max_ang_accel
        self.goal_tol = self.conf.goal_tol
        self.speed_samples = self.conf.speed_samples
        self.heading_span = self.conf.heading_span
        self.heading_samples = self.conf.heading_samples
        self.prim_horiz = self.conf.prim_horiz
        self.kinematics = self.conf.kinematics
        self.sensing_dist = self.conf.sensing_dist
        self.heading = helper.angle(self.goal - self.start)
        self.speed = self.max_speed
        # self.speed = 0
        self.vel = self.speed * helper.vec(self.heading)
        self.speeds = np.linspace(self.max_speed, self.min_speed, self.speed_samples)
        self.rel_headings = np.linspace(
            -self.heading_span / 2, self.heading_span / 2, self.heading_samples
        )
        self.rel_prims = self.prim_horiz * np.multiply.outer(
            self.speeds, helper.vec(self.rel_headings)
        )
        self.pos = self.start.copy()
        self.patches = Patches()
        self.pos_log = np.full(
            (int(self.env_conf.max_duration / self.env_conf.dt) + 1, 2), np.inf
        )
        self.heading_log = np.full(
            int(self.env_conf.max_duration / self.env_conf.dt) + 1, np.inf
        )
        self.speed_log = np.full(
            int(self.env_conf.max_duration / self.env_conf.dt) + 1, np.inf
        )
        self.vel_log = np.full(
            (int(self.env_conf.max_duration / self.env_conf.dt) + 1, 2), np.inf
        )
        self.col_log = np.full(int(self.env_conf.max_duration / self.env_conf.dt) + 1, False)
        self.goal_log = np.full(int(self.env_conf.max_duration / self.env_conf.dt) + 1, False)
        self.past_vels = self.vel * np.ones((2, 2))
        self.collided = False
        self.update_abs_prims()
        self.update_abs_headings()
        self.goal_check()

    def __repr__(self):
        # return f'Agent {self.id} {self.policy}'
        # return (f"Agent {self.id}: policy={self.policy}, " \
        #         f"start=[{self.start[0]:.2f}, {self.start[1]:.2f}], " \
        #         f"goal=[{self.goal[0]:.2f}, {self.goal[1]:.2f}]")
        return (
            f"Agent {self.id}: policy={self.policy}, "
            + f"[[{self.start[0]:.2f}, {self.start[1]:.2f}], "
            + f"[{self.goal[0]:.2f}, {self.goal[1]:.2f}]], "
            + f"max_speed={self.max_speed:.2f}"
        )

    def post_init(self):
        self.update_agent_list()

    def update_agent_list(self):
        self.other_agents = {id: a for id, a in self.env.agents.items() if id != self.id}

    def goal_check(self):
        self.at_goal = helper.dist(self.pos, self.goal) <= self.goal_tol
        if self.at_goal and not hasattr(self, "time_to_goal"):
            self.time_to_goal = self.env.time

    def collision_check(self):
        for a in self.other_agents.values():
            self.collided |= helper.dist(self.pos, a.pos) <= 2 * self.conf.radius

    def update_abs_prims(self):
        self.abs_prims = self.pos + helper.rotate(self.rel_prims, self.heading)

    def update_abs_headings(self):
        self.abs_headings = helper.wrap_to_pi(self.heading + self.rel_headings)

    def get_action(self):
        self.des_speed = self.max_speed
        self.des_heading = helper.angle(self.goal - self.pos)

    def step(self):
        if not self.at_goal and not self.collided:
            if self.kinematics == "first_order_unicycle":
                self.speed = self.des_speed
                self.heading = self.des_heading
            elif self.kinematics == "second_order_unicycle":
                self.speed += self.env.dt * np.clip(
                    (self.des_speed - self.speed) / self.env.dt,
                    -self.max_accel,
                    self.max_accel,
                )
                self.heading += self.env.dt * np.clip(
                    helper.wrap_to_pi(self.des_heading - self.heading) / self.env.dt,
                    -self.max_ang_accel,
                    self.max_ang_accel,
                )
                self.heading = helper.wrap_to_pi(self.heading)
            else:
                raise NotImplementedError
            self.vel = self.speed * helper.vec(self.heading)
            self.pos += self.env.dt * self.vel
        else:
            self.speed = 0
            self.vel = np.array([0, 0])

    def log_data(self, step):
        self.pos_log[step] = self.pos
        self.heading_log[step] = self.heading
        self.past_vels = np.roll(self.past_vels, 1, axis=0)
        self.past_vels[0] = self.vel
        self.vel_log[step] = self.vel
        self.speed_log[step] = self.speed
