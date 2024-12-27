from algos import Algo
from baseagent import Agent, Command


class DebugSarsa(Algo):
    """
    Parameters
    ----------
    alpha : float [0, 1]
        Step-size.
    gamma : float [0, 1]
        Discount-rate.
    """
    def __init__(self, alpha=0.1, gamma=0.9):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal):
        ret = self.alpha * (r + self.gamma * agent.q(new_s, new_a) - agent.q(s, a))
        return Command(ret, s, a, is_terminal)


class DebugQLearn(Algo):
    """
    Parameters
    ----------
    alpha : float [0, 1]
        Step-size.
    gamma : float [0, 1]
        Discount-rate.
    """
    def __init__(self, alpha=0.1, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, agent: Agent , s, a, r, new_s, new_a, t, is_terminal):
        best = agent.best_action_val(new_s)
        ret = self.alpha * (r + self.gamma * best - agent.q(s, a))
        return Command(ret, s, a, is_terminal)
