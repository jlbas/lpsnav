from dataclasses import dataclass
import itertools
import logging
import numpy as np
from utils import helper


@dataclass
class AgentObs:
    radius: float

    def get_agent_obs(self, agent):
        self.pos = agent.pos.copy()
        self.heading = agent.heading
        self.speed = agent.speed
        self.vel = agent.vel.copy()


@dataclass
class Logs:
    def __init__(self, cnt, val):
        self.pos = np.full((cnt, 2), val)
        self.heading = np.full(cnt, val)
        self.speed = np.full(cnt, val)
        self.vel = np.full((cnt, 2), val)


class Env:
    def __init__(self, conf, agents):
        self.dt = conf["dt"]
        self.max_duration = conf["max_duration"]
        self.agents = agents
        self.ego_id = list(agents.values())[0].id
        self.max_step = int(self.max_duration / self.dt)
        self.time = 0
        self.step = 0
        self.logger = logging.getLogger(__name__)
        self.logs = {k: Logs(self.max_step + 1, np.nan) for k in self.agents}
        self.log_data()
        self.agent_obs = {k: AgentObs(a.radius) for k, a in self.agents.items()}
        for k, a in self.agents.items():
            self.agent_obs[k].get_agent_obs(a)
        for a in self.agents.values():
            self.logger.debug(a)
            a.post_init(self.dt, self.agent_obs)

    def log_data(self):
        for k, log in self.logs.items():
            for attr in vars(log):
                getattr(log, attr)[self.step] = getattr(self.agents[k], attr)

    def sense_agents(self, a):
        agents = {}
        for neighbour_id, neighbour in self.agent_obs.items():
            if a.id != neighbour_id:
                if helper.dist(a.pos, neighbour.pos) < a.sensing_dist:
                    agents[neighbour_id] = neighbour
        return agents

    def update(self):
        for a in self.agents.values():
            agent_obs = self.sense_agents(a)
            a.get_action(self.dt, agent_obs)
        for a in self.agents.values():
            a.step(self.dt)
        self.time += self.dt
        self.step += 1
        self.collision_check()
        for a in [a for a in self.agents.values() if not hasattr(a, "ttg")]:
            a.goal_check(self.time)
        self.log_data()
        for k, a in self.agents.items():
            self.agent_obs[k].get_agent_obs(a)

    def collision_check(self):
        for a1, a2 in itertools.combinations(self.agents.values(), 2):
            collided = helper.dist(a1.pos, a2.pos) <= a1.radius + a2.radius
            a1.collided |= collided
            a2.collided |= collided

    def is_running(self):
        if all([hasattr(a, "ttg") for a in self.agents.values()]):
            msg = f"Simulation ended at {self.time:.2f}s. All agents reached their goals."
        elif all([hasattr(a, "ttg") or a.collided for a in self.agents.values()]):
            msg = f"Simulation ended at {self.time:.2f}s. Some agents have collided."
        elif self.step >= self.max_step:
            msg = "Simulation time limit was reached. Not all agents reached their goals."
        else:
            return True
        self.logger.debug(msg)
        return False

    def trim_logs(self):
        for log in self.logs.values():
            for k in vars(log):
                setattr(log, k, getattr(log, k)[: self.step + 1])
        self.time_log = np.around(np.linspace(0, self.time, self.step + 1), 3)
