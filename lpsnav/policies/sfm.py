import numpy as np
import sympy as smp
from policies.agent import Agent
from utils import helper


class Sfm(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.tau = conf["tau"]
        self.sigma = conf["sigma"]
        self.v_ab0 = conf["v_ab0"]
        self.phi = conf["phi"]
        self.c = conf["c"]
        self.step_width = conf["step_width"]
        self.u_ab0 = conf["u_ab0"]
        self.R = conf["R"]
        self.goal_force = np.zeros(2)
        self.ped_force = np.zeros(2)
        self.tot_force = np.zeros(2)
        self.init_gradient()

    def init_gradient(self):
        r_abx, r_aby, v_b, dt, e_bx, e_by, sigma, v_ab0 = smp.symbols(
            "r_abx r_aby v_b dt e_bx e_by sigma, v_ab0"
        )
        arg_11 = smp.sqrt(r_abx**2 + r_aby**2)
        arg_12 = smp.sqrt((r_abx - v_b * dt * e_bx) ** 2 + (r_aby - v_b * dt * e_by) ** 2)
        arg_2 = v_b * dt
        b = 0.5 * smp.sqrt((arg_11 + arg_12) ** 2 - arg_2**2)
        v_ab = v_ab0 * smp.exp(-b / sigma)
        self.grad_v_ab = smp.lambdify(
            (r_abx, r_aby, v_b, dt, e_bx, e_by, sigma, v_ab0), [smp.diff(v_ab, r_abx), smp.diff(v_ab, r_aby)]
        )
        u_ab = self.u_ab0 * smp.exp(-arg_11 / self.R)
        self.grad_u_ab = smp.lambdify((r_abx, r_aby), [smp.diff(u_ab, r_abx), smp.diff(u_ab, r_aby)])

    def dir_weight(self, e, f):
        return 1 if np.dot(e, f) >= np.linalg.norm(f) * np.cos(self.phi) else self.c

    def get_goal_force(self):
        self.e_a = helper.unit_vec(self.goal - self.pos)
        self.goal_force = (self.max_speed * self.e_a - self.vel) / self.tau

    def get_ped_force(self, agents):
        self.ped_force = np.zeros(2)
        for a in agents.values():
            r_ab_cc = self.pos - a.pos
            d = helper.dist(self.pos, a.pos)
            scale = (d - 2 * self.radius) / d
            r_ab = scale * r_ab_cc
            e_b = helper.vec(a.heading)
            f_ab = -np.array(
                self.grad_v_ab(*r_ab, a.speed, self.step_width, *e_b, self.sigma, self.v_ab0)
            )
            w = self.dir_weight(self.e_a, -f_ab)
            self.ped_force += w * f_ab
            self.ped_force = np.nan_to_num(self.ped_force)

    def get_action(self, dt, agents):
        self.get_goal_force()
        self.get_ped_force(agents)
        self.tot_force = self.goal_force + self.ped_force
        des_vel = helper.clip(self.vel + self.tot_force * dt, self.max_speed)
        self.des_speed = np.linalg.norm(des_vel)
        self.des_heading = helper.wrap_to_pi(helper.angle(des_vel))
