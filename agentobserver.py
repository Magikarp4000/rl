from baseagent import Agent


class AgentObserver:
    def __init__(self):
        self.agent = None

    def observe(self, agent: Agent):
        self.agent = agent
