import numpy as np
from env.agent_factory import init_agents


class Env:
    def __init__(self, config, ego_policy, iter, scenario, policy_id=0):
        self.config = config
        self.timestep = self.config.timestep
        self.max_duration = self.config.max_duration
        self.done = False
        self.time = 0
        self.step = 0
        self.ego_policy = ego_policy
        self.iter = iter
        self.scenario = scenario
        self.policy_id = policy_id
        self.ego_agent, self.agents = init_agents(
            self.config, self, self.ego_policy, self.scenario, self.policy_id
        )
        for agent in self.agents.values():
            print(agent)
            agent.post_init()
            agent.log_data(self.step)

    def __str__(self):
        ret = f"{self.scenario}_{self.ego_policy}"
        if self.scenario == "random" or self.scenario == "circle":
            ret += f"_{len(self.agents)}"
        if self.scenario == "random":
            ret += f"_{self.iter}"
        return ret

    def update(self):
        for a in self.agents.values():
            a.goal_check()
        for a in self.agents.values():
            if not a.at_goal and not a.collided:
                a.get_action()
        for a in self.agents.values():
            a.step()
        self.time += self.timestep
        self.step += 1
        for a in self.agents.values():
            a.collision_check()
            if not a.collided:
                a.log_data(self.step)
        self.check_if_done()

    def check_if_done(self):
        if all([a.at_goal for a in self.agents.values()]):
            print(f"Simulation ended at {self.time:.2f}s. All agents reached their goals.")
            self.done = True
        elif all([a.at_goal or a.collided for a in self.agents.values()]):
            print(f"Simulation ended at {self.time:.2f}s. Some agents have collided.")
            self.done = True
        elif self.time >= self.max_duration - self.timestep:
            print(
                f"Simulation duration of {self.max_duration}s was reached. "
                "Not all agents reached their goals."
            )
            self.done = True
        if self.done:
            print(60 * "=")

    def trim_logs(self):
        for a in self.agents.values():
            max_idx = np.argmax(np.all(a.pos_log == [np.inf, np.inf], axis=-1))
            if max_idx:
                a.pos_log = a.pos_log[:max_idx]
                a.heading_log = a.heading_log[:max_idx]
