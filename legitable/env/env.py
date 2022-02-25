import logging
import numpy as np
from env.agent_factory import init_agents


class Env:
    def __init__(self, config, rng, ego_policy, iter, scenario, n_agents, ws_len, sg_ws_ratio, policy_id=0):
        self.config = config
        self.dt = self.config.env.dt
        self.max_step = int(self.config.env.max_duration / self.dt)
        self.done = False
        self.time = 0
        self.step = 0
        self.ego_policy = ego_policy
        self.iter = iter
        self.scenario = scenario
        self.n_agents = n_agents
        self.policy_id = policy_id
        self.ego_agent, self.agents = init_agents(
            self.config, self, rng, self.ego_policy, self.scenario, self.n_agents, ws_len, sg_ws_ratio, self.policy_id
        )
        self.logger = logging.getLogger(__name__)
        for agent in self.agents.values():
            self.logger.debug(agent)
            agent.post_init()
            agent.log_data(self.step)

    def __str__(self):
        ret = [self.scenario, self.ego_policy]
        if self.n_agents != -1:
            ret.append(self.n_agents)
        if self.scenario == "random":
            ret.append(self.iter)
        return '_'.join(map(str, ret))

    def update(self):
        for a in [a for a in self.agents.values() if not a.collided]:
            if hasattr(a, "ttg"):
                a.des_speed = 0
            else:
                a.get_action()
        for a in self.agents.values():
            a.step()
        self.time += self.dt
        self.step += 1
        for a in self.agents.values():
            not hasattr(a, "ttg") and a.goal_check()
            not a.collided and a.collision_check()
            a.log_data(self.step)
        self.check_if_done()

    def check_if_done(self):
        if hasattr(self.ego_agent, "ttg") and not self.config.env.homogeneous:
            self.logger.debug(f"Simulation ended at {self.time:.2f}s. Ego agent reached its goal.")
        elif all([hasattr(a, "ttg") for a in self.agents.values()]) and self.config.env.homogeneous:
            self.logger.debug(f"Simulation ended at {self.time:.2f}s. All agents reached their goals.")
        elif all([hasattr(a, "ttg") or a.collided for a in self.agents.values()]):
            self.logger.debug(f"Simulation ended at {self.time:.2f}s. Some agents have collided.")
        elif self.step >= self.max_step:
            self.logger.debug("Simulation time limit was reached. Not all agents reached their goals.")
        else:
            return
        self.done = True

    def trim_logs(self):
        max_idx = np.argmax(np.any(~np.isfinite(self.ego_agent.pos_log), axis=-1))
        if max_idx:
            for a in self.agents.values():
                a.pos_log = a.pos_log[:max_idx]
                a.heading_log = a.heading_log[:max_idx]
                a.vel_log = a.vel_log[:max_idx]
                a.speed_log = a.speed_log[:max_idx]
