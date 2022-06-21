import numpy as np
from utils import helper


class Agent:
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, _rng):
        self.id = id
        self.name = conf["name"]
        self.color = conf["color"]
        self.policy = policy
        self.is_ego = is_ego
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.eps = conf["eps"]
        self.radius = conf["radius"]
        r = 0.9 * self.radius
        x = r * np.cos(np.pi / 6)
        y = r * np.sin(np.pi / 6)
        self.body_coords = [(r, 0), (-x, y), (-x, -y)]
        self.min_speed = conf["min_speed"]
        self.max_speed = max_speed
        self.max_accel = conf["max_accel"]
        self.min_accel = conf["min_accel"]
        self.max_ang_accel = conf["max_ang_accel"]
        self.goal_tol = conf["goal_tol"]
        self.speed_samples = conf["speed_samples"]
        self.heading_span = conf["heading_span"]
        self.heading_samples = conf["heading_samples"]
        self.prim_horiz = conf["prim_horiz"]
        self.kinematics = conf["kinematics"]
        self.sensing_dist = conf["sensing_dist"]
        self.agent_col_horiz = conf["agent_col_horiz"]
        self.wall_col_horiz = conf["wall_col_horiz"]
        self.heading = helper.angle(self.goal - self.start)
        self.speed = np.clip(conf.get("init_speed", self.max_speed), self.min_speed, self.max_speed)
        self.vel = self.speed * helper.vec(self.heading)
        self.speeds = np.linspace(self.max_speed, self.min_speed, self.speed_samples)
        self.rel_headings = np.linspace(
            -self.heading_span / 2, self.heading_span / 2, self.heading_samples
        )
        self.rel_prims = self.prim_horiz * np.multiply.outer(
            self.speeds, helper.vec(self.rel_headings)
        )
        self.pos = self.start.copy()
        self.collided = False
        self.update_abs_prims()
        self.update_abs_headings()
        self.update_abs_prim_vels()

    def __repr__(self):
        return (
            f"Agent {self.id}: policy={self.policy}, "
            + f"[[{self.start[0]:.2f}, {self.start[1]:.2f}], "
            + f"[{self.goal[0]:.2f}, {self.goal[1]:.2f}]], "
            + f"max_speed={self.max_speed:.2f}"
        )

    def post_init(self, _dt, _agents, _walls):
        pass

    def goal_check(self, time):
        if helper.dist(self.pos, self.goal) <= self.goal_tol:
            self.ttg = time - getattr(self, "start_time", 0)

    def update_abs_prims(self):
        self.abs_prims = self.pos + helper.rotate(self.rel_prims, self.heading)

    def update_abs_headings(self):
        self.abs_headings = helper.wrap_to_pi(self.heading + self.rel_headings)

    def update_abs_prim_vels(self):
        self.abs_prim_vels = np.multiply.outer(self.speeds, helper.vec(self.abs_headings))

    def remove_col_prims(self, dt, agents, walls):
        self.col_mask = np.full((self.speed_samples, self.heading_samples), False)
        goal_dist = helper.dist(self.pos, self.goal)
        buffer = 0.2 * self.speed / self.max_speed
        for t in np.linspace(0, self.agent_col_horiz, int(0.5 * self.agent_col_horiz / dt)):
            ego_pred = self.pos + t * self.abs_prim_vels
            for a in [a for a in agents.values() if helper.dist(self.pos, a.pos) < 1.2 * goal_dist]:
                a_pred = a.pos + t * a.vel
                self.col_mask |= helper.dist(ego_pred, a_pred) < self.radius + a.radius + buffer
        for t in np.linspace(0, self.wall_col_horiz, int(0.5 * self.wall_col_horiz / dt)):
            ego_pred = self.pos + t * self.abs_prim_vels
            for wall in walls:
                self.col_mask |= helper.dist_to_line_seg(ego_pred, *wall) < self.radius + buffer

    def get_action(self, _dt, _agents, _walls):
        self.des_speed = self.max_speed
        self.des_heading = helper.angle(self.goal - self.pos)

    def step(self, dt):
        if not self.collided:
            if hasattr(self, "ttg"):
                self.des_speed = 0
                self.des_heading = self.heading
            if self.kinematics == "first_order_unicycle":
                self.speed = self.des_speed
                self.heading = self.des_heading
            elif self.kinematics == "second_order_unicycle":
                self.speed += dt * np.clip(
                    (self.des_speed - self.speed) / dt,
                    self.min_accel,
                    self.max_accel,
                )
                self.heading += dt * np.clip(
                    helper.wrap_to_pi(self.des_heading - self.heading) / dt,
                    -self.max_ang_accel,
                    self.max_ang_accel,
                )
                self.heading = helper.wrap_to_pi(self.heading)
            else:
                raise NotImplementedError
            self.vel = self.speed * helper.vec(self.heading)
            self.pos += dt * self.vel
        else:
            self.speed = 0
            self.vel = np.zeros(2)
