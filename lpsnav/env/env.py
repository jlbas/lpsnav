from dataclasses import dataclass
import itertools
import logging
import numpy as np
from utils import helper


@dataclass
class ObsAgent:
    radius: float

    def update(self, agent):
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
        self.logs = {id: Logs(self.max_step + 1, np.nan) for id in self.agents}
        self.log_data()
        self.obs_agents = {id: ObsAgent(a.radius) for id, a in self.agents.items()}
        for id, a in self.agents.items():
            self.obs_agents[id].update(a)
        for a in self.agents.values():
            self.logger.debug(a)
            a.post_init(self.dt, self.obs_agents)

    def log_data(self):
        for id, log in self.logs.items():
            for k in vars(log):
                getattr(log, k)[self.step] = getattr(self.agents[id], k)

    def update(self):
        for id1, a1 in self.agents.items():
            in_range = lambda a2: helper.dist(a1.pos, a2.pos) < a1.sensing_dist
            agents = {id2: a2 for id2, a2 in self.obs_agents.items() if id2 != id1 and in_range(a2)}
            a1.get_action(self.dt, agents)
        for a1 in self.agents.values():
            a1.step(self.dt)
        self.time += self.dt
        self.step += 1
        self.collision_check()
        for a1 in [a for a in self.agents.values() if not hasattr(a, "ttg")]:
            a1.goal_check(self.time)
        self.log_data()
        for id, a1 in self.agents.items():
            self.obs_agents[id].update(a1)

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
