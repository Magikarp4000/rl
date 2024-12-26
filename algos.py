from abc import ABC, abstractmethod
import inspect

import numpy as np

from baseagent import Agent, Command
from utils import Buffer


class Algo(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_params(self):
        args = inspect.getfullargspec(self.__init__)[0][1:]
        params = {'algo': self.name} | {arg: getattr(self, arg) for arg in args}
        return params

    @abstractmethod
    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal): pass
    def init_episode(self, s, a): pass


class Sarsa(Algo):
    """
    Parameters
    ----------
    alpha : float [0, 1], default=0.1
        Step-size.
    gamma : float [0, 1], default=0.9
        Discount-rate.
    """
    def __init__(self, alpha=0.1, gamma=0.9):
        super().__init__('sarsa')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal):
        ret = self.alpha * (r + self.gamma * agent.q(new_s, new_a) - agent.q(s, a))
        return Command(ret, s, a, is_terminal)


class QLearn(Algo):
    """
    Parameters
    ----------
    alpha : float [0, 1], default=0.1
        Step-size.
    gamma : float [0, 1], default=0.9
        Discount-rate.
    """
    def __init__(self, alpha=0.1, gamma=0.9):
        super().__init__('qlearn')
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, agent: Agent , s, a, r, new_s, new_a, t, is_terminal):
        best = agent.best_action_val(new_s)
        ret = self.alpha * (r + self.gamma * best - agent.q(s, a))
        return Command(ret, s, a, is_terminal)


class NStepAlgo(Algo):
    """
    Parameters
    ----------
    alpha : float [0, 1], default=0.1
        Step-size.
    gamma : float [0, 1], default=0.9
        Discount-rate.
    nstep : int [1, inf), default=5
        Number of bootstrapped steps used in updates.
    """
    def __init__(self, alpha=0.1, gamma=0.9, nstep=5):
        super().__init__('nstepsarsa')
        self.alpha = alpha
        self.gamma = gamma
        self.nstep = nstep

        self.buffer = Buffer(nstep + 1, (0, 0, 0))
        self.T_step = None
    
    def init_episode(self, s, a):
        self.buffer.reset()
        self.buffer.set(0, (s, a, 0))
        self.T_step = np.inf

    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal):
        if self.T_step is np.inf and is_terminal:
            self.T_step = t

        if t <= self.T_step:
            self.buffer.set(t + 1, (new_s, new_a, r))
        
        tgt_t = t - self.nstep + 1
        if tgt_t < 0:
            return Command(no_update=True)

        tgt_s, tgt_a = self.buffer.get(tgt_t)[:2]
        ret = self.alpha * (self.get_return(agent, t, tgt_t) - agent.q(tgt_s, tgt_a))
        terminate = tgt_t >= self.T_step

        return Command(ret, tgt_s, tgt_a, terminate)

    def get_return(self, agent: Agent, t, tgt_t):
        end_t = min(t + 1, self.T_step + 1)
        ret = self.init_return(agent, *self.buffer.get(end_t))
        for i in range(end_t - 1, tgt_t, -1):
            ret = self.step_return(agent, *self.buffer.get(i), ret)
        return ret
    
    @abstractmethod
    def init_return(self, agent, s, a, r): pass
    @abstractmethod
    def step_return(self, agent, s, a, r, ret): pass


class NStepSarsa(NStepAlgo):
    def init_return(self, agent, s, a, r):
        return r + self.gamma * agent.q(s, a)
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class NStepExpectedSarsa(NStepAlgo):
    def init_return(self, agent, s, a, r):
        return r + self.gamma * sum([agent.action_prob(s, a) * agent.q(s, a)])
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class NStepQLearn(NStepAlgo):
    def init_return(self, agent, s, a, r):
        return r + self.gamma * agent.best_action_val(s)
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class TreeLearn(NStepAlgo):
    def init_return(self, agent, s, a, r):
        return r + self.gamma * agent.best_action_val(s)
    
    def step_return(self, agent, s, a, r, ret):
        best_a = agent.best_action(s)
        if a == best_a:
            return r + self.gamma * ret
        else:
            return r + self.gamma * agent.q(s, best_a)
