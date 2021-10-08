from policies.agent import Agent

class Inattentive(Agent):

    def __init__(self, config, env, id, policy, start, goal):
        super().__init__(config, env, id, policy, start, goal)
        self.color = "gray"
