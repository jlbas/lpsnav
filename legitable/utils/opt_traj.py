import casadi as cs
import numpy as np
from utils import helper


def rotate(vec, th):
    if vec.shape[0] != 2:
        x, y = cs.vertsplit(vec.T)
    else:
        x, y = cs.vertsplit(vec)
    x_rot = x * cs.cos(th) - y * cs.sin(th)
    y_rot = x * cs.sin(th) + y * cs.cos(th)
    return cs.vertcat(x_rot, y_rot)


def dynamics(x, u):
    return cs.vertcat(
        x[3, :] * cs.cos(x[2, :]), x[3, :] * cs.sin(x[2, :]), x[4, :], u[0, :], u[1, :]
    )


def cost_to_pt(pos_0, speed_0, pos_1, vel_1):
    d_vec = pos_0 - pos_1
    d = cs.sqrt(cs.sum2(d_vec ** 2))
    d_vec_hat = d_vec / d
    rot_d_vec_hat = cs.horzcat(-d_vec_hat[:, 1], d_vec_hat[:, 0])
    speed_1_t = cs.sqrt(cs.sum2(rot_d_vec_hat * cs.repmat(vel_1.T, rot_d_vec_hat.shape[0])) ** 2)
    speed_1_r = cs.sum2(d_vec_hat * cs.repmat(vel_1.T, d_vec_hat.shape[0]))
    arg = 1 - (speed_1_t / speed_0) ** 2
    arg *= arg >= 0
    speed_0_r = speed_0 * cs.sqrt(arg)
    den = speed_0_r + speed_1_r
    den *= den > 0
    den += (den == 0) * 1e-5
    return d / den


def cost_to_line(pt, pt_speed, line_0, line_1, line_vel):
    vec = line_1 - line_0
    v0 = cs.horzcat(-vec[:, 1], vec[:, 0])
    v1 = -v0
    r = pt - line_0
    v = cs.sum2(v0 * r)
    v = v0 * (v > 0) + v1 * (v <= 0)
    v_hat = v / cs.sqrt(cs.sum2(v ** 2))
    d = cs.sqrt(cs.sum2(r * v_hat) ** 2)
    proj_line_speed = cs.sum2(v_hat * cs.repmat(line_vel.T, v_hat.shape[0]))
    den = pt_speed + proj_line_speed
    den *= den > 0
    den += (den == 0) * 1e-5
    return d / den


def p_intersect(pos, v, line_0, line_1, line_th, pt_vel, t_to_line):
    th = cs.arctan2(cs.sin(line_th + np.pi), cs.cos(line_th + np.pi))
    vel = v * cs.horzcat(cs.cos(th), cs.sin(th))
    r_pred = pos + vel * t_to_line
    p0_pred = line_0 + cs.repmat(pt_vel.T, t_to_line.shape[0]) * t_to_line
    p1_pred = line_1 + cs.repmat(pt_vel.T, t_to_line.shape[0]) * t_to_line
    w = r_pred - p0_pred
    v = p1_pred - p0_pred
    c0 = cs.sum2(w * v)
    c1 = cs.sum2(v * v)
    return cs.horzcat(c0 <= 0, (c0 > 0) * (c1 > c0), c1 <= c0)


def masked_cost(masks, cost_col_0, cost_col_1, cost_line):
    left = masks[:, 0] * cs.horzcat(cost_line, cost_col_0, cost_col_1)
    center = masks[:, 1] * cs.horzcat(cost_col_0, cost_line, cost_col_1)
    right = masks[:, 2] * cs.horzcat(cost_col_0, cost_col_1, cost_line)
    return left + center + right


def dynamic_pt_cost(pt, pt_speed, line_0, line_1, line_th, line_vel):
    cost_col_0 = cost_to_pt(pt, pt_speed, line_0, line_vel)
    cost_col_1 = cost_to_pt(pt, pt_speed, line_1, line_vel)
    cost_line = cost_to_line(pt, pt_speed, line_0, line_1, line_vel)
    masks = p_intersect(pt, pt_speed, line_0, line_1, line_th, line_vel, cost_line)
    return masked_cost(masks, cost_col_0, cost_col_1, cost_line)


def get_opt_traj(int_idx, int_slice, passing_idx, dt, ego_agent, agent, receding_horiz, priors):
    opti = cs.Opti()
    v_max = max(ego_agent.max_speed, agent.max_speed + 0.1)

    cw = ego_agent.radius + agent.radius
    base_line = np.array([[0, -cw], [0, cw]])
    line_heading = helper.wrap_to_pi(helper.angle(ego_agent.pos_log[int_idx[0]] - ego_agent.goal))
    int_line = helper.rotate(base_line, line_heading) + agent.pos_log[int_idx[0]]
    min_t = helper.cost_to_line(
        ego_agent.pos_log[int_idx[0]],
        v_max,
        int_line,
        agent.vel_log[int_idx[0]],
    )

    N = int(np.diff(int_idx))
    t = cs.linspace(0, N * dt, N + 1)

    X = opti.variable(5, N + 1)
    U = opti.variable(2, N)

    x, y, th, v, w = cs.vertsplit(X)
    opti.subject_to(x[0] == ego_agent.pos_log[int_idx[0]][0])
    opti.subject_to(y[0] == ego_agent.pos_log[int_idx[0]][1])
    opti.subject_to(th[0] == ego_agent.heading_log[int_idx[0]])
    opti.subject_to(opti.bounded(0, v, v_max))
    opti.set_initial(
        x, np.linspace(ego_agent.pos_log[int_idx[0]][0], ego_agent.pos_log[int_idx[1]][0], N + 1)
    )
    opti.set_initial(
        y, np.linspace(ego_agent.pos_log[int_idx[0]][1], ego_agent.pos_log[int_idx[1]][1], N + 1)
    )
    opti.set_initial(th, ego_agent.heading_log[int_idx[0]])
    opti.set_initial(v, v_max)
    opti.set_initial(w, 0)
    opti.set_initial(U, 1)

    for i in range(N):
        k1 = dynamics(X[:, i], U[:, i])
        k2 = dynamics(X[:, i] + k1 * dt / 2, U[:, i])
        k3 = dynamics(X[:, i] + k2 * dt / 2, U[:, i])
        k4 = dynamics(X[:, i] + k3 * dt, U[:, i])
        x_next = X[:, i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        opti.subject_to(X[:, i + 1] == x_next)

    goal_vec = cs.repmat(ego_agent.goal, 1, N + 1).T - cs.vertcat(x, y).T
    line_heading = cs.arctan2(goal_vec[:, 1], goal_vec[:, 0])
    rel_line_0 = cs.horzcat(cw * cs.sin(line_heading), -cw * cs.cos(line_heading))
    rel_line_1 = cs.horzcat(-cw * cs.sin(line_heading), cw * cs.cos(line_heading))
    agent_pos = (
        cs.repmat(agent.pos_log[int_idx[0]], 1, N + 1).T
        + t * cs.repmat(agent.vel_log[int_idx[0]], 1, N + 1).T
    )
    agent_pos = agent.pos_log[int_slice]
    int_line_0 = rel_line_0 + agent_pos
    int_line_1 = rel_line_1 + agent_pos
    ego_init_pos = cs.MX(ego_agent.pos_log[int_idx[0]])
    ego_init_vel = cs.MX(ego_agent.vel_log[int_idx[0]])
    receding_steps = int(receding_horiz / dt)
    init_receded_pos = cs.repmat(ego_init_pos.T, receding_steps) - dt * cs.linspace(
        receding_steps, 1, receding_steps
    ) * cs.repmat(ego_init_vel.T, receding_steps)
    receded_pos = cs.vertcat(init_receded_pos, cs.vertcat(x, y)[:, :-receding_steps].T)[: N + 1, :]
    agent_init_vel = cs.MX(agent.vel_log[int_idx[0]])
    receded_line_0 = int_line_0 - cs.repmat(agent_init_vel.T, N + 1) * receding_horiz
    receded_line_1 = int_line_1 - cs.repmat(agent_init_vel.T, N + 1) * receding_horiz

    cost_rg = dynamic_pt_cost(
        receded_pos, v_max, receded_line_0, receded_line_1, line_heading, agent_init_vel
    )
    cost_tg = dynamic_pt_cost(
        cs.vertcat(x, y).T, v_max, int_line_0, int_line_1, line_heading, agent_init_vel
    )
    cost_rtg = receding_horiz + cost_tg
    arg = cost_rg - cost_rtg
    vals = cs.exp(arg) * cs.repmat(priors, 1, N + 1).T
    vals /= cs.sum2(vals)
    d_vals = t[::-1] * vals
    num = dt * (cs.sum1(d_vals[1:-1, :]) + 0.5 * (d_vals[0, :] + d_vals[-1, :]))
    den = dt * (cs.sum1(t[1:-1]) + 0.5 * (t[0] + t[-1]))
    leg_score = num / den

    p_opts = {"ipopt.print_level": 0, "print_time": 0}
    s_opts = {"max_iter": 1e2}
    opti.solver("ipopt", p_opts, s_opts)

    min_sols = []
    max_sols = []
    try:
        opti.minimize(-leg_score[0])
        max_left_sol = opti.solve()
        max_left = max_left_sol.value(leg_score[0])
        max_sols.append(max_left)
    except RuntimeError:
        print("Couldn't find solution for max left")

    try:
        opti.minimize(leg_score[0])
        min_left_sol = opti.solve()
        min_left = min_left_sol.value(leg_score[0])
        min_sols.append(min_left)
    except RuntimeError:
        print("Couldn't find solution for min left")

    try:
        opti.minimize(-leg_score[2])
        max_right_sol = opti.solve()
        max_right = max_right_sol.value(leg_score[2])
        max_sols.append(max_right)
    except RuntimeError:
        print("Couldn't find solution for max right")

    try:
        opti.minimize(leg_score[2])
        min_right_sol = opti.solve()
        min_right = min_right_sol.value(leg_score[2])
        min_sols.append(min_right)
    except RuntimeError:
        print("Couldn't find solution for min right")

    return (None, None) if not min_sols or not max_sols else (min(min_sols), max(max_sols))
