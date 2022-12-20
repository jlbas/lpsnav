import numpy as np
import pickle
import copy

from policies.agent import Agent
from utils import helper
import math


class Neural_network_regr_multi:
    def __init__(self, filename):
        with open(filename, "rb") as fo:
            nn_list = pickle.load(fo, encoding="latin1")
        self.W = nn_list[0]
        self.b = nn_list[1]
        self.avg_vec = nn_list[2]
        self.std_vec = nn_list[3]
        self.output_avg_vec = nn_list[4]
        self.output_std_vec = nn_list[5]
        self.layers_info = nn_list[6]
        self.layers_type = nn_list[7]
        self.num_hidden_layers = len(self.W) - 1
        self.layers_dim = []
        for i in range(len(self.layers_info)):
            self.layers_dim.append(
                int(np.sum(self.layers_info[i][:, 0] * self.layers_info[i][:, 1]))
            )
        return

    def xRaw_2_x(self, X_raw):
        if len(X_raw.shape) > 1:
            nb_examples = X_raw.shape[0]
        else:
            nb_examples = 1
        X = (X_raw - np.tile(self.avg_vec, (nb_examples, 1))) / np.tile(
            self.std_vec, (nb_examples, 1)
        )
        return X

    def y_2_yRaw(self, Y):
        if len(Y.shape) > 1:
            nb_examples = Y.shape[0]
        else:
            nb_examples = 1
        Y_raw = Y * np.tile(self.output_std_vec, (nb_examples, 1)) + np.tile(
            self.output_avg_vec, (nb_examples, 1)
        )
        return Y_raw

    def make_prediction(self, X):
        if len(X.shape) > 1:
            nb_examples = X.shape[0]
        else:
            nb_examples = 1
            X = X[np.newaxis, :]

        nb_layers = self.num_hidden_layers + 1
        out = X
        for layer in range(nb_layers - 1):
            if self.layers_type[layer] == "conn":
                tmp = np.dot(out, self.W[layer]) + np.tile(self.b[layer], (nb_examples, 1))
                out = tmp * (tmp > 0)
            elif self.layers_type[layer] == "max":
                num_pts = out.shape[0]
                next_layer_size = np.sum(self.layers_info[layer][:, 1])
                out_next = np.zeros((num_pts, np.sum(next_layer_size)))
                if num_pts == 0:
                    out = out_next
                    continue
                cur_s_ind = 0
                next_s_ind = 0
                for ii in range(self.layers_info[layer].shape[0]):
                    num_agents = self.layers_info[layer][ii, 0]
                    stride = self.layers_info[layer][ii, 1]
                    cur_e_ind = cur_s_ind + num_agents * stride
                    next_e_ind = next_s_ind + stride
                    block_form = np.reshape(out[:, cur_s_ind:cur_e_ind], (num_pts, -1, stride))
                    out_next[:, next_s_ind:next_e_ind] = np.max(block_form, axis=1)
                    cur_s_ind = cur_e_ind
                    next_s_ind = next_e_ind

                out = out_next

        y_hat = np.dot(out, self.W[nb_layers - 1]) + np.tile(
            self.b[nb_layers - 1], (nb_examples, 1)
        )
        return y_hat

    def make_prediction_raw(self, X_raw):
        X = self.xRaw_2_x(X_raw)
        Y_scale = self.make_prediction(X)
        Y_hat = self.y_2_yRaw(Y_scale)
        return Y_hat


def find_dist_between_segs(x1, x2, y1, y2):
    if_one_pt = False
    if x2.shape == (2,):
        x2 = x2.reshape((1, 2))
        y2 = y2.reshape((1, 2))
        if_one_pt = True
    end_dist = np.linalg.norm(x2 - y2, axis=1)
    critical_dist = end_dist.copy()
    z_bar = (x2 - x1) - (y2 - y1)
    inds = np.where((np.linalg.norm(z_bar, axis=1) > 0))[0]
    t_bar = -np.sum((x1 - y1) * z_bar[inds, :], axis=1) / np.sum(
        z_bar[inds, :] * z_bar[inds, :], axis=1
    )
    t_bar_rep = np.tile(t_bar, (2, 1)).transpose()
    dist_bar = np.linalg.norm(
        x1 + (x2[inds, :] - x1) * t_bar_rep - y1 - (y2[inds, :] - y1) * t_bar_rep,
        axis=1,
    )
    inds_2 = np.where((t_bar > 0) & (t_bar < 1.0))
    critical_dist[inds[inds_2]] = dist_bar[inds_2]

    min_dist = np.amin(np.vstack((end_dist, critical_dist)), axis=0)

    if if_one_pt:
        return min_dist[0]
    else:
        return min_dist


def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff


def reorder_other_agents_state(agent_state, others_state):
    num_agents = len(others_state)
    dist_2_others = np.zeros((num_agents))
    for i, other_state in enumerate(others_state):
        dist_2_others[i] = np.linalg.norm(other_state[0:2] - agent_state[0:2])
    closest_ind = np.argmin(dist_2_others)
    others_state_cp = copy.deepcopy(others_state)
    others_state_cp[0] = others_state[closest_ind]
    others_state_cp[closest_ind] = others_state[0]
    return others_state_cp


def rawState_2_agentCentricState(agent_state, others_state_in, num_agents_in_network, eps):
    others_state = reorder_other_agents_state(agent_state, others_state_in)
    num_agents = len(others_state) + 1
    try:
        assert num_agents <= num_agents_in_network
    except AssertionError:
        print("num_agents, num_agents_in_network", num_agents, num_agents_in_network)
        assert 0
    state_nn = np.zeros((7 + 8 * (num_agents_in_network - 1),))
    for i in range(num_agents - 1, num_agents_in_network - 1):
        state_nn[7 + 8 * i : 7 + 8 * i + 7] = [-2.0, -2.0, -10, -10.0, -0.2, -0.2, -2.0]

    goal_direction = agent_state[6:8] - agent_state[0:2]
    dist_to_goal = np.clip(np.linalg.norm(goal_direction), 0, 30)
    pref_speed = agent_state[5]
    if dist_to_goal > eps:
        ref_prll = goal_direction / dist_to_goal
    else:
        ref_prll = np.array([np.cos(agent_state[4]), np.sin(agent_state[4])])
    ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg

    ref_prll_angle = np.arctan2(ref_prll[1], ref_prll[0])
    heading = find_angle_diff(agent_state[4], ref_prll_angle)
    heading_cos = np.cos(heading)
    heading_sin = np.sin(heading)

    cur_speed = np.linalg.norm(agent_state[2:4])

    vx = cur_speed * heading_cos
    vy = cur_speed * heading_sin
    self_radius = agent_state[8]

    state_nn[0:7] = [dist_to_goal, pref_speed, cur_speed, heading, vx, vy, self_radius]

    for i, other_agent_state in enumerate(others_state):
        rel_pos = other_agent_state[0:2] - agent_state[0:2]
        rel_pos_x = np.clip(np.dot(rel_pos, ref_prll), -8, 8)
        rel_pos_y = np.clip(np.dot(rel_pos, ref_orth), -8, 8)
        other_vx = np.dot(other_agent_state[2:4], ref_prll)
        other_vy = np.dot(other_agent_state[2:4], ref_orth)
        other_radius = other_agent_state[8]
        dist_2_other = np.clip(
            np.linalg.norm(agent_state[0:2] - other_agent_state[0:2]) - self_radius - other_radius,
            -3,
            10,
        )
        is_on = 1
        if other_vx**2 + other_vy**2 < eps:
            is_on = 2
        state_nn[7 + 8 * i : 7 + 8 * (i + 1)] = [
            other_vx,
            other_vy,
            rel_pos_x,
            rel_pos_y,
            other_radius,
            self_radius + other_radius,
            dist_2_other,
            is_on,
        ]

    for i in range(num_agents - 1, num_agents_in_network - 1):
        state_nn[7 + 8 * i : 7 + 8 * (i + 1) - 1] = state_nn[7 : 7 + 8 - 1]

    return ref_prll, ref_orth, state_nn


def rawStates_2_agentCentricStates(agent_states, others_states_in, num_agents_in_network, eps):
    if agent_states.shape[0] >= 1:
        others_states = reorder_other_agents_state(agent_states[0, :], others_states_in)
    else:
        others_states = others_states_in
    num_agents = len(others_states) + 1
    assert num_agents <= num_agents_in_network
    num_rawStates = agent_states.shape[0]
    states_nn = np.zeros((num_rawStates, 7 + 8 * (num_agents_in_network - 1)))
    for i in range(num_agents - 1, num_agents_in_network - 1):
        states_nn[:, 7 + 8 * i : 7 + 8 * i + 7] = np.tile(
            np.array([-2.0, -2.0, -10, -10.0, -0.2, -0.2, -2.0]), (num_rawStates, 1)
        )
    goal_direction = agent_states[:, 6:8] - agent_states[:, 0:2]
    dist_to_goal = np.clip(np.linalg.norm(goal_direction, axis=1), 0, 30)
    pref_speed = agent_states[:, 5]

    valid_inds = np.where(dist_to_goal > eps)[0]
    ref_prll = np.vstack([np.cos(agent_states[:, 4]), np.sin(agent_states[:, 4])]).transpose()
    ref_prll[valid_inds, 0] = goal_direction[valid_inds, 0] / dist_to_goal[valid_inds]
    ref_prll[valid_inds, 1] = goal_direction[valid_inds, 1] / dist_to_goal[valid_inds]

    ref_orth = np.vstack([-ref_prll[:, 1], ref_prll[:, 0]]).transpose()

    ref_prll_angle = np.arctan2(ref_prll[:, 1], ref_prll[:, 0])
    heading = find_angle_diff(agent_states[:, 4], ref_prll_angle)
    heading_cos = np.cos(heading)
    heading_sin = np.sin(heading)

    cur_speed = np.linalg.norm(agent_states[:, 2:4], axis=1)

    vx = cur_speed * heading_cos
    vy = cur_speed * heading_sin
    self_radius = agent_states[:, 8]

    states_nn[:, 0:7] = np.vstack(
        (dist_to_goal, pref_speed, cur_speed, heading, vx, vy, self_radius)
    ).transpose()

    for i, other_agent_states in enumerate(others_states):
        rel_pos = other_agent_states[0:2] - agent_states[:, 0:2]
        rel_pos_x = np.clip(np.sum(rel_pos * ref_prll, axis=1), -8, 8)
        rel_pos_y = np.clip(np.sum(rel_pos * ref_orth, axis=1), -8, 8)
        other_vx = np.sum(other_agent_states[2:4] * ref_prll, axis=1)
        other_vy = np.sum(other_agent_states[2:4] * ref_orth, axis=1)
        other_radius = other_agent_states[8] * np.ones((num_rawStates,))
        is_on = np.ones((num_rawStates,))
        stat_inds = np.where(other_vx**2 + other_vy**2 < eps)[0]
        is_on[stat_inds] = 2

        dist_2_other = np.clip(
            np.linalg.norm(agent_states[:, 0:2] - other_agent_states[0:2], axis=1)
            - self_radius
            - other_radius,
            -3,
            10,
        )
        states_nn[:, 7 + 8 * i : 7 + 8 * (i + 1)] = np.vstack(
            (
                other_vx,
                other_vy,
                rel_pos_x,
                rel_pos_y,
                other_radius,
                self_radius + other_radius,
                dist_2_other,
                is_on,
            )
        ).transpose()

    for i in range(num_agents - 1, num_agents_in_network - 1):
        states_nn[:, 7 + 8 * i : 7 + 8 * (i + 1) - 1] = states_nn[:, 7 : 7 + 8 - 1]

    return ref_prll, ref_orth, states_nn


class NN_navigation_value:
    def __init__(self, filename, num_agents):
        self.num_agents = num_agents
        self.nn = Neural_network_regr_multi(filename)
        self.dt_forward = 1.0
        self.radius_buffer = 0.0

    def find_actions_theta(self, agent_state, default_action_theta):
        num_near_actions = 10
        zero_action = np.zeros((1, 2))
        desired_act = np.array(
            [
                agent_state[5],
                np.arctan2(agent_state[7] - agent_state[1], agent_state[6] - agent_state[0]),
            ]
        )
        desired_actions = np.tile(desired_act, (5, 1))
        desired_actions[1, 0] *= 0.80
        desired_actions[2, 0] *= 0.60
        desired_actions[3, 0] *= 0.40
        desired_actions[4, 0] *= 0.20

        tmp_action_theta = default_action_theta.copy()
        tmp_action_theta[0] = agent_state[5]
        near_actions = np.tile(tmp_action_theta, (num_near_actions, 1))
        near_actions[:, 1] += np.linspace(-np.pi / 3.0, np.pi / 3.0, num=num_near_actions)

        near_actions_reduced = near_actions.copy()
        near_actions_reduced_1 = near_actions.copy()
        near_actions_reduced_2 = near_actions.copy()
        near_actions_reduced[:, 0] *= 0.75
        near_actions_reduced_1[:, 0] *= 0.50
        near_actions_reduced_2[:, 0] *= 0.25

        near_actions = np.vstack(
            (
                near_actions,
                near_actions_reduced,
                near_actions_reduced_1,
                near_actions_reduced_2,
            )
        )

        actions = np.vstack((default_action_theta, desired_actions, zero_action, near_actions))

        actions[:, 1] = (actions[:, 1] + np.pi) % (np.pi * 2) - np.pi

        return actions

    def find_action_rewards(
        self,
        agent_state,
        cur_dist,
        min_dists,
        getting_close_range,
        collision_cost,
        gamma,
        dt_normal,
    ):
        rewards = np.zeros((len(min_dists),))
        if cur_dist < 0:
            rewards[:] = collision_cost
            return rewards

        d = np.linalg.norm(agent_state[0:2] - agent_state[6:8])
        v = agent_state[5]
        getting_close_penalty = gamma ** (d / dt_normal) * (1.0 - gamma ** (-v / dt_normal))

        close_inds = np.where((min_dists > 0) & (min_dists < getting_close_range))[0]

        if cur_dist < getting_close_range:
            assert getting_close_range - cur_dist > 0
            rewards[:] = getting_close_penalty

        rewards[close_inds] += getting_close_penalty

        collision_inds = np.where(min_dists < 0)[0]
        rewards[collision_inds] = collision_cost

        scaling_cur = 2
        scaling_future = 5
        rewards[close_inds] = scaling_cur * rewards[
            close_inds
        ] + scaling_future * getting_close_penalty * (getting_close_range - min_dists[close_inds])

        rewards[close_inds] = np.clip(rewards[close_inds], collision_cost + 0.01, 0.0)
        assert np.all(getting_close_range - min_dists[close_inds] > 0)

        return rewards

    def check_collisions_and_get_action_rewards(
        self,
        agent_state,
        actions_theta,
        other_agents_state_in,
        other_agents_action,
        dt_forward,
        eps,
        getting_close_range,
        collision_cost,
        gamma,
        dt_normal,
    ):
        other_agents_state = copy.deepcopy(other_agents_state_in)

        num_actions = actions_theta.shape[0]
        num_other_agents = len(other_agents_state)
        for tt in range(num_other_agents):
            other_agents_state[tt][2] = other_agents_action[tt][0] * np.cos(
                other_agents_action[tt][1]
            )
            other_agents_state[tt][3] = other_agents_action[tt][0] * np.sin(
                other_agents_action[tt][1]
            )

        other_agents_next_state = []
        for tt in range(num_other_agents):
            other_agents_next_state.append(
                self.update_state(other_agents_state[tt], other_agents_action[tt], dt_forward)
            )

        min_dists_mat = np.zeros((num_actions, num_other_agents))
        if_collide_mat = np.zeros((num_actions, num_other_agents))
        cur_dist_vec = np.zeros((num_other_agents,))
        for tt in range(num_other_agents):
            min_dists_mat[:, tt], if_collide_mat[:, tt] = self.if_actions_collide(
                agent_state,
                actions_theta,
                other_agents_state[tt],
                other_agents_action[tt],
                dt_forward,
                eps,
                getting_close_range,
            )

            radius = agent_state[8] + other_agents_state[tt][8] + self.radius_buffer
            cur_dist_vec[tt] = (
                np.linalg.norm(agent_state[0:2] - other_agents_state[tt][0:2]) - radius
            )

        min_dists = np.min(min_dists_mat, axis=1)
        if_collide = np.max(if_collide_mat, axis=1)
        cur_dist = np.min(cur_dist_vec)

        action_rewards = self.find_action_rewards(
            agent_state, cur_dist, min_dists, getting_close_range, collision_cost, gamma, dt_normal
        )

        return (
            if_collide,
            action_rewards,
            min_dists,
            other_agents_next_state,
            num_actions,
        )

    def find_values_and_action_rewards(
        self,
        agent_state,
        actions_theta,
        other_agents_state_in,
        other_agents_action,
        dt_forward,
        eps,
        getting_close_range,
        collision_cost,
        gamma,
        dt_normal,
        dist_2_goal_thres,
    ):
        (
            if_collide,
            action_rewards,
            min_dists,
            other_agents_next_state,
            num_actions,
        ) = self.check_collisions_and_get_action_rewards(
            agent_state,
            actions_theta,
            other_agents_state_in,
            other_agents_action,
            dt_forward,
            eps,
            getting_close_range,
            collision_cost,
            gamma,
            dt_normal,
        )
        state_values = np.zeros((num_actions,))
        non_collision_inds = np.where(if_collide == False)[0]

        gamma = gamma
        dt_normal = dt_normal
        if len(non_collision_inds) > 0:
            agent_next_states = self.update_states(
                agent_state, actions_theta[non_collision_inds, :], dt_forward
            )

            dists_to_goal = np.linalg.norm(
                agent_next_states[:, 0:2] - agent_next_states[:, 6:8], axis=1
            )

            reached_goals_inds = np.where(
                (dists_to_goal < dist_2_goal_thres)
                & (min_dists[non_collision_inds] > getting_close_range)
            )[0]
            not_reached_goals_inds = np.setdiff1d(
                np.arange(len(non_collision_inds)), reached_goals_inds
            )

            non_collision_reached_goals_inds = non_collision_inds[reached_goals_inds]
            non_collision_not_reached_goals_inds = non_collision_inds[not_reached_goals_inds]

            state_values[non_collision_not_reached_goals_inds] = self.find_states_values(
                agent_next_states[not_reached_goals_inds],
                other_agents_next_state,
                eps,
                gamma,
                dt_normal,
            )

            state_values[non_collision_reached_goals_inds] = gamma ** (
                dists_to_goal[reached_goals_inds] / dt_normal
            )

        return state_values, action_rewards

    def find_next_states_values(
        self,
        agent_state,
        actions_theta,
        other_agents_state,
        other_agents_action,
        eps,
        getting_close_range,
        collision_cost,
        gamma,
        dt_normal,
        dist_2_goal_thres,
    ):
        values, _, _ = self.find_next_states_values_and_components(
            agent_state,
            actions_theta,
            other_agents_state,
            other_agents_action,
            eps,
            getting_close_range,
            collision_cost,
            gamma,
            dt_normal,
            dist_2_goal_thres,
        )
        return values

    def find_next_states_values_and_components(
        self,
        agent_state,
        actions_theta,
        other_agents_state,
        other_agents_action,
        eps,
        getting_close_range,
        collision_cost,
        gamma,
        dt_normal,
        dist_2_goal_thres,
    ):
        agent_speed = agent_state[5]
        dt_forward_max = max(self.dt_forward, 0.5 / agent_speed)
        dist_to_goal = np.linalg.norm(agent_state[6:8] - agent_state[0:2])
        time_to_goal = dist_to_goal / agent_speed
        dt_forward = min(dt_forward_max, time_to_goal)  # 1.0

        state_values, action_rewards = self.find_values_and_action_rewards(
            agent_state,
            actions_theta,
            other_agents_state,
            other_agents_action,
            dt_forward,
            eps,
            getting_close_range,
            collision_cost,
            gamma,
            dt_normal,
            dist_2_goal_thres,
        )

        gamma = gamma
        dt_normal = dt_normal
        agent_desired_speed = agent_state[5]

        num_states = len(actions_theta)
        dt_forward_vec = 0.2 * np.ones((num_states,)) * dt_forward
        dt_forward_vec += 0.8 * actions_theta[:, 0] / agent_desired_speed * dt_forward
        values = (
            action_rewards
            + gamma ** (dt_forward_vec * agent_desired_speed / dt_normal) * state_values
        )

        return values, state_values, action_rewards

    def find_feasible_actions(self, agent_state):
        default_action_xy = agent_state[2:4]
        speed = np.linalg.norm(default_action_xy)
        angle_select = agent_state[4]
        default_action_theta = np.array([speed, angle_select])
        actions_theta = self.find_actions_theta(agent_state, default_action_theta)
        return actions_theta

    def find_next_action(
        self,
        agent_state,
        other_agents_state,
        other_agents_actions,
        eps,
        getting_close_range,
        collision_cost,
        gamma,
        dt_normal,
        dist_2_goal_thres,
    ):
        actions_theta = self.find_feasible_actions(agent_state)
        state_values = self.find_next_states_values(
            agent_state,
            actions_theta,
            other_agents_state,
            other_agents_actions,
            eps,
            getting_close_range,
            collision_cost,
            gamma,
            dt_normal,
            dist_2_goal_thres,
        )
        best_action_ind = np.argmax(state_values)
        best_action = actions_theta[best_action_ind]
        return best_action

    def update_state(self, state, action_theta, dt):
        speed = action_theta[0]
        angle_select = action_theta[1]
        next_state = copy.deepcopy(state)
        pref_speed = state[5]
        next_state[0] += speed * np.cos(angle_select) * dt
        next_state[1] += speed * np.sin(angle_select) * dt
        next_state[2] = speed * np.cos(angle_select)
        next_state[3] = speed * np.sin(angle_select)
        next_state[5] = pref_speed
        next_state[4] = angle_select
        return next_state

    def update_states(self, state, actions_theta, dt):
        speeds = actions_theta[:, 0]
        angles_select = actions_theta[:, 1]
        num_actions = actions_theta.shape[0]
        next_states = np.tile(state, (num_actions, 1))
        pref_speed = state[5]
        next_states[:, 0] += speeds * np.cos(angles_select) * dt
        next_states[:, 1] += speeds * np.sin(angles_select) * dt
        next_states[:, 2] = speeds * np.cos(angles_select)
        next_states[:, 3] = speeds * np.sin(angles_select)
        next_states[:, 4] = angles_select
        next_states[:, 5] = pref_speed

        return next_states

    def if_actions_collide(
        self,
        agent_state,
        agent_actions,
        other_agent_state,
        other_agent_action,
        delta_t,
        eps,
        getting_close_range,
    ):
        agent_pref_speed = agent_state[5]
        other_agent_speed = other_agent_action[0]
        radius = agent_state[8] + other_agent_state[8] + self.radius_buffer
        num_actions = agent_actions.shape[0]
        if_collide = np.zeros((num_actions), dtype=bool)
        min_dists = (radius + getting_close_range + eps) * np.ones((num_actions), dtype=bool)
        if (
            np.linalg.norm(agent_state[0:2] - other_agent_state[0:2])
            > (agent_pref_speed + other_agent_speed) * delta_t + radius
        ):
            return min_dists, if_collide

        agent_vels = np.zeros((num_actions, 2))
        agent_vels[:, 0] = agent_actions[:, 0] * np.cos(agent_actions[:, 1])
        agent_vels[:, 1] = agent_actions[:, 0] * np.sin(agent_actions[:, 1])
        other_agent_v = np.zeros((2,))
        other_agent_v[0] = other_agent_action[0] * np.cos(other_agent_action[1])
        other_agent_v[1] = other_agent_action[0] * np.sin(other_agent_action[1])
        other_agent_vels = np.tile(other_agent_v, (num_actions, 1))

        p_oa_angle = np.arctan2(
            other_agent_state[1] - agent_state[1], other_agent_state[0] - agent_state[0]
        )
        agent_speed_angles = np.arctan2(agent_vels[:, 1], agent_vels[:, 0])
        other_speed_angle = np.arctan2(other_agent_v[1], other_agent_v[0])
        heading_diff = find_angle_diff(agent_speed_angles, other_speed_angle)
        agent_heading_2_other = find_angle_diff(agent_speed_angles, p_oa_angle)
        r = agent_state[8] + other_agent_state[8] + getting_close_range
        coll_angle = abs(
            np.arcsin(min(0.95, r / np.linalg.norm(agent_state[0:2] - other_agent_state[0:2])))
        )

        front_inds = np.where(
            (abs(agent_heading_2_other) < coll_angle) & (abs(heading_diff) < np.pi / 2.0)
        )[0]
        if len(front_inds) > 0:
            dot_product = np.sum(agent_vels * other_agent_vels, axis=1)
            valid_inds = np.where(agent_vels[:, 0] > eps)[0]
            dot_product[valid_inds] /= np.linalg.norm(agent_vels[valid_inds, :], axis=1)
            other_agent_vels[front_inds, :] = (
                other_agent_vels[front_inds, :]
                - np.tile(dot_product[front_inds], (2, 1)).transpose()
                * agent_vels[front_inds, :]
                / 2.0
            )

        x1 = agent_state[0:2]
        x2 = x1 + min(1.0, delta_t) * agent_vels
        y1 = other_agent_state[0:2]
        y2 = y1 + min(1.0, delta_t) * other_agent_vels

        min_dists = find_dist_between_segs(x1, x2, y1, y2)

        cur_dist = np.linalg.norm(x1 - y1)
        if cur_dist < radius:
            if_collide[:] = True
        else:
            if_collide = min_dists < radius
        min_dists = min_dists - radius
        return min_dists, if_collide

    def find_states_values(self, agent_states, other_agents_state, eps, gamma, dt_normal):
        if agent_states.ndim == 1:
            _, _, state_nn = rawState_2_agentCentricState(
                agent_states, other_agents_state, self.num_agents, eps
            )

            value = np.squeeze(self.nn.make_prediction_raw(state_nn).clip(min=-0.25, max=1.0))
            upper_bnd = gamma ** (state_nn[0] / dt_normal)
            value = min(upper_bnd, value)

            return value
        else:
            _, _, states_nn = rawStates_2_agentCentricStates(
                agent_states, other_agents_state, self.num_agents, eps
            )

            values = np.squeeze(self.nn.make_prediction_raw(states_nn).clip(min=-0.25, max=1.0))
            upper_bnd = gamma ** (states_nn[:, 0] / dt_normal)
            values = np.minimum(upper_bnd, values)

            return values


def filter_vel(dt_vec, agent_past_vel_xy):
    average_x = np.sum(dt_vec * agent_past_vel_xy[:, 0]) / np.sum(dt_vec)
    average_y = np.sum(dt_vec * agent_past_vel_xy[:, 1]) / np.sum(dt_vec)
    speeds = np.linalg.norm(agent_past_vel_xy, axis=1)
    speed = np.linalg.norm(np.array([average_x, average_y]))
    angle = np.arctan2(average_y, average_x)
    return np.array([speed, angle])


def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class SaCadrl(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.eps = conf["eps"]
        self.getting_close_range = conf["getting_close_range"]
        self.collision_cost = conf["collision_cost"]
        self.gamma = conf["gamma"]
        self.dt_normal = conf["dt_normal"]
        self.dist_2_goal_thres = conf["dist_2_goal_thres"]
        fname = "policies/drl_checkpoints/4_agents_policy_iter_1000.p"
        self.value_net = NN_navigation_value(fname, 4)
        self.update_ego_frame()

    def post_init(self, _dt, agents):
        self.vel_hist = {k: a.vel * np.ones((2, 2)) for k, a in agents.items()}

    def get_action(self, dt, agents):
        self.des_speed, d_heading = self.find_next_action(dt, agents)
        self.des_heading = helper.wrap_to_pi(self.heading + d_heading)
        for k, a in agents.items():
            self.vel_hist[k] = np.roll(self.vel_hist[k], 1, axis=0)
            self.vel_hist[k][0] = a.vel

    def step(self, dt):
        super().step(dt)
        self.update_ego_frame()

    def find_next_action(self, dt, agents):
        agent_state, other_agents_state, other_agents_actions = self.parse_agents(dt, agents)
        action = self.query_and_rescale_action(
            agent_state, other_agents_state, other_agents_actions
        )
        return action

    def parse_agents(self, dt, agents):
        agent_state = self.convert_host_agent_to_cadrl_state()
        (
            other_agents_state,
            other_agents_actions,
        ) = self.convert_other_agents_to_cadrl_state(dt, agents)
        return agent_state, other_agents_state, other_agents_actions

    def convert_host_agent_to_cadrl_state(self):
        x, y = self.pos
        v_x, v_y = self.vel
        radius = self.radius
        heading_angle = self.heading
        pref_speed = self.max_speed
        goal_x, goal_y = self.goal

        agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, goal_x, goal_y, radius])

        return agent_state

    def convert_other_agents_to_cadrl_state(self, dt, agents):
        other_agent_dists = []
        for k, a in agents.items():
            rel_pos_to_other_global_frame = a.pos - self.pos
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_orth)
            dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - self.radius - a.radius
            other_agent_dists.append([k, round(dist_2_other, 2), p_orthog_ego_frame])
        sorted_dists = sorted(other_agent_dists, key=lambda x: (-x[1], x[2]))
        sorted_inds = [x[0] for x in sorted_dists]
        clipped_sorted_inds = sorted_inds[-3:]
        clipped_sorted_agents = {k: agents[k] for k in clipped_sorted_inds}

        agents = clipped_sorted_agents

        other_agents_state = []
        other_agents_actions = []
        for k, a in agents.items():
            x, y = a.pos
            v_x, v_y = a.vel
            radius = a.radius
            heading_angle = np.nan # hidden state (unused by CADRL)
            pref_speed = np.nan # hidden state (unused by CADRL)
            goal_x, goal_y = (np.nan, np.nan) # hidden state (unused by CADRL)

            past_vel = self.vel_hist[k]
            dt_past_vec = dt * np.ones((2))
            filtered_actions_theta = filter_vel(dt_past_vec, past_vel)
            other_agents_actions.append(filtered_actions_theta)

            other_agent_state = np.array(
                [x, y, v_x, v_y, heading_angle, pref_speed, goal_x, goal_y, radius]
            )
            other_agents_state.append(other_agent_state)
        return other_agents_state, other_agents_actions

    def update_ego_frame(self):
        self.ref_prll, self.ref_orth = self.get_ref()
        ref_prll_angle_global_frame = np.arctan2(self.ref_prll[1], self.ref_prll[0])
        self.heading_ego_frame = wrap(self.heading - ref_prll_angle_global_frame)

    def get_ref(self):
        goal_direction = self.goal - self.pos
        self.dist_to_goal = math.sqrt(goal_direction[0] ** 2 + goal_direction[1] ** 2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])
        return ref_prll, ref_orth

    def query_and_rescale_action(self, agent_state, other_agents_state, other_agents_actions):
        if len(other_agents_state) > 0:
            action = self.value_net.find_next_action(
                agent_state,
                other_agents_state,
                other_agents_actions,
                self.eps,
                self.getting_close_range,
                self.collision_cost,
                self.gamma,
                self.dt_normal,
                self.dist_2_goal_thres,
            )
            action[1] = wrap(action[1] - self.heading)
        else:
            action = np.array([1.0, -self.heading_ego_frame])
        return action
