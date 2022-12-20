import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from policies.agent import Agent
from utils import helper
import math


class Actions:
    def __init__(self):
        self.actions = (
            np.mgrid[1.0:1.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 12].reshape(2, -1).T
        )
        self.actions = np.vstack(
            [
                self.actions,
                np.mgrid[0.5:0.6:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T,
            ]
        )
        self.actions = np.vstack(
            [
                self.actions,
                np.mgrid[0.0:0.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6].reshape(2, -1).T,
            ]
        )
        self.num_actions = len(self.actions)


class NetworkVPCore(object):
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

    def crop_x(self, x):
        if x.shape[-1] > self.x.shape[-1]:
            x_ = x[:, : self.x.shape[-1]]
        elif x.shape[-1] < self.x.shape[-1]:
            x_ = np.zeros((x.shape[0], self.x.shape[-1]))
            x_[:, : x.shape[1]] = x
        else:
            x_ = x
        return x_

    def predict_p(self, x):
        x = self.crop_x(x)
        return self.sess.run(self.softmax_p, feed_dict={self.x: x})

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
            raise NotImplementedError
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):

                self.sess = tf.compat.v1.Session(
                    graph=self.graph,
                    config=tf.compat.v1.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
                    ),
                )

                new_saver = tf.compat.v1.train.import_meta_graph(
                    filename + ".meta", clear_devices=True
                )
                self.sess.run(tf.compat.v1.global_variables_initializer())
                new_saver.restore(self.sess, filename)

                self.softmax_p = g.get_tensor_by_name("Softmax:0")
                self.x = g.get_tensor_by_name("X:0")
                self.v = g.get_tensor_by_name("Squeeze:0")


def vec2_l2_norm(vec):
    return math.sqrt(vec2_l2_norm_squared(vec))


def vec2_l2_norm_squared(vec):
    return vec[0] ** 2 + vec[1] ** 2


class Ga3cCadrl(Agent):
    def __init__(self, conf, id, policy, is_ego, max_speed, start, goal, rng):
        super().__init__(conf, id, policy, is_ego, max_speed, start, goal, rng)
        self.max_num_agents_in_environment = 19
        self.max_num_other_agents_observed = 18
        self.other_agent_states = np.zeros((7,))
        self.possible_actions = Actions()
        self.device = "/cpu:0"
        self.nn = NetworkVPCore(self.device, "network", self.possible_actions.num_actions)
        filename = "policies/drl_checkpoints/network_01900000"
        self.nn.simple_load(filename)
        self.update_ego_frame()

    def get_action(self, _dt, agents):
        self.action = self.find_next_action(agents)
        self.des_speed = self.action[0]
        self.des_heading = helper.wrap_to_pi(self.heading + self.action[1])

    def step(self, dt):
        super().step(dt)
        self.update_ego_frame()

    def find_next_action(self, agents):
        self.get_obs(agents)
        predictions = self.nn.predict_p(self.vec_obs)[0]
        action_index = np.argmax(predictions)
        raw_action = self.possible_actions.actions[action_index]
        action = np.array([self.max_speed * raw_action[0], raw_action[1]])
        return action

    def get_obs(self, agents):
        self.sense(agents)
        num_other_agents = self.num_other_agents_observed
        dist_to_goal = helper.dist(self.pos, self.goal)
        heading_ego_frame = self.heading_ego_frame
        pref_speed = self.max_speed
        radius = self.radius
        other_agents_states = self.other_agents_states.flatten()
        self.vec_obs = np.hstack(
            (
                np.array(
                    [
                        num_other_agents,
                        dist_to_goal,
                        heading_ego_frame,
                        pref_speed,
                        radius,
                    ]
                ),
                other_agents_states,
            )
        )[None, :]

    def sense(self, agents):
        sorting_criteria = []
        for i, other_agent in agents.items():
            rel_pos_to_other_global_frame = other_agent.pos - self.pos
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_orth)
            dist_between_agent_centers = vec2_l2_norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - self.radius - other_agent.radius
            combined_radius = self.radius + other_agent.radius
            time_to_impact = None
            sorting_criteria.append([i, round(dist_2_other, 2), p_orthog_ego_frame, time_to_impact])
        clipped_sorted_inds = self.get_clipped_sorted_inds(sorting_criteria)
        clipped_sorted_agents = [agents[i] for i in clipped_sorted_inds]

        self.other_agents_states = np.zeros((self.max_num_other_agents_observed, 7))
        other_agent_count = 0
        for other_agent in clipped_sorted_agents:
            rel_pos_to_other_global_frame = other_agent.pos - self.pos
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_orth)
            v_parallel_ego_frame = np.dot(other_agent.vel, self.ref_prll)
            v_orthog_ego_frame = np.dot(other_agent.vel, self.ref_orth)
            dist_2_other = (
                np.linalg.norm(rel_pos_to_other_global_frame) - self.radius - other_agent.radius
            )
            combined_radius = self.radius + other_agent.radius

            other_obs = np.array(
                [
                    p_parallel_ego_frame,
                    p_orthog_ego_frame,
                    v_parallel_ego_frame,
                    v_orthog_ego_frame,
                    other_agent.radius,
                    combined_radius,
                    dist_2_other,
                ]
            )

            if other_agent_count == 0:
                self.other_agent_states[:] = other_obs

            self.other_agents_states[other_agent_count, :] = other_obs
            other_agent_count += 1

        self.num_other_agents_observed = other_agent_count

    def get_clipped_sorted_inds(self, sorting_criteria):
        sorted_sorting_criteria = sorted(sorting_criteria, key=lambda x: (x[1], x[2]))
        clipped_sorting_criteria = sorted_sorting_criteria[: self.max_num_agents_in_environment - 1]
        sorted_dists = sorted(clipped_sorting_criteria, key=lambda x: (x[1], x[2]))
        clipped_sorted_inds = [x[0] for x in sorted_dists]
        return clipped_sorted_inds

    def parse_agents(self):
        agent_state = self.convert_host_agent_to_cadrl_state()
        (
            other_agents_state,
            other_agents_actions,
        ) = self.convert_other_agents_to_cadrl_state()
        return agent_state, other_agents_state, other_agents_actions

    def update_ego_frame(self):
        self.ref_prll, self.ref_orth = self.get_ref()
        ref_prll_angle_global_frame = np.arctan2(self.ref_prll[1], self.ref_prll[0])
        self.heading_ego_frame = helper.wrap_to_pi(self.heading - ref_prll_angle_global_frame)

    def get_ref(self):
        goal_direction = self.goal - self.pos
        self.dist_to_goal = math.sqrt(goal_direction[0] ** 2 + goal_direction[1] ** 2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg
        return ref_prll, ref_orth

    def query_and_rescale_action(self, agent_state, other_agents_state, other_agents_actions):
        if len(other_agents_state) > 0:
            action = self.value_net.find_next_action(
                agent_state, other_agents_state, other_agents_actions
            )
            action[1] = helper.wrap_to_pi(action[1] - self.heading)
        else:
            action = np.array([1.0, -self.heading_ego_frame])
        return action
