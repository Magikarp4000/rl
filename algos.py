from abc import ABC, abstractmethod
import baseagent


class Algo(ABC):
    @abstractmethod
    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a, *args, **kwargs):
        pass


class Sarsa(Algo):
    """
    Parameters
    ----------
    alpha : float, default=0.1
        Step-size.
    gamma : float, default=0.9
        Discount-rate.
    """
    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a,
                 alpha=0.1,
                 gamma=0.9):
        return alpha * (r + gamma * agent.q(new_s, new_a) - agent.q(s, a))


class Qlearn(Algo):
    """
    Parameters
    ----------
    alpha : float, default=0.1
        Step-size.
    gamma : float, default=0.9
        Discount-rate.
    """
    def __call__(self, agent: baseagent.Agent , s, a, r, new_s, new_a,
                 alpha=0.1,
                 gamma=0.9):
        best = max([agent.q(new_s, next_a) for next_a in agent.env.action_spec(s)])
        return alpha * (r + gamma * best - agent.q(s, a))