from abc import ABC, abstractmethod
import baseagent


class Algo(ABC):
    @abstractmethod
    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a, *args, **kwargs):
        pass


class Sarsa(Algo):
    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a,
                 gamma=0.9,
                 alpha=0.1):
        return alpha * (r + gamma * agent.q(new_s, new_a) - agent.q(s, a))


class Qlearn(Algo):
    def __call__(self, agent: baseagent.Agent , s, a, r, new_s, new_a,
                 gamma=0.9,
                 alpha=0.1):
        best = max([agent.q(new_s, next_a) for next_a in agent.env.action_spec(s)])
        return alpha * (r + gamma * best - agent.q(s, a))