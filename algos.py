from abc import ABC, abstractmethod
import inspect
import baseagent


class Algo(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_params(self):
        args = inspect.getfullargspec(self.__init__)[0][1:]
        params = {'algo': self.name} | {arg: getattr(self, arg) for arg in args}
        return params

    @abstractmethod
    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a, step, *args, **kwargs):
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
    def __init__(self, alpha=0.1, gamma=0.9):
        super().__init__('sarsa')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a, step):
        return self.alpha * (r + self.gamma * agent.q(new_s, new_a) - agent.q(s, a))


class Qlearn(Algo):
    """
    Parameters
    ----------
    alpha : float, default=0.1
        Step-size.
    gamma : float, default=0.9
        Discount-rate.
    """
    def __init__(self, alpha=0.1, gamma=0.9):
        super().__init__('qlearn')
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, agent: baseagent.Agent , s, a, r, new_s, new_a, step):
        best = max([agent.q(new_s, next_a) for next_a in agent.env.action_spec(new_s)])
        return self.alpha * (r + self.gamma * best - agent.q(s, a))


class NStepSarsa(Algo):
    def __init__(self, alpha=0.1, gamma=0.9, nstep=1):
        super().__init__('nstepsarsa')
        self.alpha = alpha
        self.gamma = gamma
        self.nstep = nstep

        self.cache = [None for _ in range(nstep)]

    def __call__(self, agent: baseagent.Agent, s, a, r, new_s, new_a, step):
        pass