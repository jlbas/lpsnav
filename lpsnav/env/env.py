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
class WallObs:
    pts: list[list[float]]
    
    def get_wall_obs(self, wall):
        self.pts = wall.copy()


@dataclass
class Logs:
    def __init__(self, cnt, val):
        self.pos = np.full((cnt, 2), val)
        self.heading = np.full(cnt, val)
        self.speed = np.full(cnt, val)
        self.vel = np.full((cnt, 2), val)


class Env:
    def __init__(self, conf, agents, walls):
        self.dt = conf["dt"]
        self.max_duration = conf["max_duration"]
        self.active_range = conf["active_range"]
        self.agents = agents
        self.active_agents = []
        self.walls = walls
        self.ego_id = list(agents.values())[0].id
        self.max_step = int(self.max_duration / self.dt)
        self.time = 0
        self.step = 0
        self.logger = logging.getLogger(__name__)
        self.logs = {id: Logs(self.max_step + 1, np.nan) for id in self.agents}
        self.log_data()
        self.agent_obs = {id: AgentObs(a.radius) for id, a in self.agents.items()}
        self.wall_obs = [WallObs(wall) for wall in self.walls]
        for id, a in self.agents.items():
            self.agent_obs[id].get_agent_obs(a)
        for i, wall in enumerate(self.walls):
            self.wall_obs[i].get_wall_obs(wall)
        for a in self.agents.values():
            self.logger.debug(a)
            a.post_init(self.dt, self.agent_obs, self.wall_obs)

    def log_data(self):
        for id, log in self.logs.items():
            for k in vars(log):
                getattr(log, k)[self.step] = getattr(self.agents[id], k)

    def sense_agents(self, a):
        agents = {}
        for neighbour_id, neighbour in self.agent_obs.items():
            if a.id != neighbour_id:
                in_range = helper.dist(a.pos, neighbour.pos) < a.sensing_dist
                in_sight = not any([helper.is_intersecting(a.pos, neighbour.pos, *wall) for wall in self.walls])
                if in_range and in_sight:
                    agents[neighbour_id] = neighbour
        return agents

    def sense_walls(self, a):
        walls = []
        for wall in self.wall_obs:
            if helper.dist_to_line_seg(a.pos, *wall.pts) < a.sensing_dist:
                walls.append(wall.pts)
        return walls

    def update(self):
        old_active_agents = self.active_agents.copy()
        self.active_agents = []
        for a in self.agents.values():
            if helper.dist(a.pos, self.agents[self.ego_id].pos) <= self.active_range:
                self.active_agents.append(a)
                if not hasattr(a, "start_time"):
                    a.start_time = self.time
        for a in old_active_agents:
            if a not in self.active_agents and not a.collided:
                a.goal = a.pos
        for a in self.active_agents:
            agent_obs = self.sense_agents(a)
            wall_obs = self.sense_walls(a)
            a.get_action(self.dt, agent_obs, wall_obs)
        for a in self.active_agents:
            a.step(self.dt)
        self.time += self.dt
        self.step += 1
        self.collision_check()
        for a in [a for a in self.agents.values() if not hasattr(a, "ttg")]:
            a.goal_check(self.time)
        self.log_data()
        for id, a in self.agents.items():
            self.agent_obs[id].get_agent_obs(a)
        for i, wall in enumerate(self.walls):
            self.wall_obs[i].get_wall_obs(wall)

    def collision_check(self):
        for a1, a2 in itertools.combinations(self.agents.values(), 2):
            collided = helper.dist(a1.pos, a2.pos) <= a1.radius + a2.radius
            a1.collided |= collided
            a2.collided |= collided
        for a in self.agents.values():
            for wall in self.walls:
                if not a.collided:
                    a.collided |= (
                        helper.dist_to_line_seg(a.pos, *wall) <= a.radius
                    )

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
