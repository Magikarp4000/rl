import inspect
from baseagent import Agent, Command


class StepData:
    def __init__(self, **kwargs):
        self.update(**kwargs)
    
    def update(self, **kwargs):
        for name, val in kwargs.items():
            setattr(self, name, val)


class Extracter:
    def extract(self, agent: Agent, s, a, r, new_a, new_s, t, cmd: Command):
        return StepData(a=a, s=s, r=r, new_s=new_s, new_a=new_a, t=t, cmd=cmd)


class ExtracterPlus(Extracter):
    def extract(self, agent, s, a, r, new_a, new_s, t, cmd):
        data = super().extract(agent, s, a, r, new_a, new_s, t, cmd)
        data.update(
            avals = list(agent.bhv_action_vals(s).flat),
            actions = agent.env.actions
        )
        return data
