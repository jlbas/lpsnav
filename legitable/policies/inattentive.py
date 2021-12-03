from policies.agent import Agent


class Inattentive(Agent):
    def __init__(self, config, env, id, policy, start, goal=None, max_speed=None):
        super().__init__(config, env, id, policy, start, goal=goal, max_speed=max_speed)
        self.color = "#868293"
