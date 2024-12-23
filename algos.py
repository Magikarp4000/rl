from abc import ABC, abstractmethod
import inspect

import numpy as np

from baseagent import Agent, Command


class Buffer:
    def __init__(self, size, default=None):
        self.size = size
        self.default = default
        self._buffer = [default for _ in range(size)]
    
    def get(self, idx):
        return self._buffer[idx % self.size]
    
    def set(self, idx, val):
        self._buffer[idx % self.size] = val
    
    def reset(self):
        self._buffer = [self.default for _ in range(self.size)]


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
    alpha : float, default=0.1
        Step-size.
    gamma : float, default=0.9
        Discount-rate.
    """
    def __init__(self, alpha=0.1, gamma=0.9):
        super().__init__('sarsa')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal):
        ret = self.alpha * (r + self.gamma * agent.q(new_s, new_a) - agent.q(s, a))
        return Command(ret, s, a, is_terminal)


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
    
    def __call__(self, agent: Agent , s, a, r, new_s, new_a, t, is_terminal):
        best = max([agent.q(new_s, next_a) for next_a in agent.env.action_spec(new_s)])
        ret = self.alpha * (r + self.gamma * best - agent.q(s, a))
        return Command(ret, s, a, is_terminal)


class NStepSarsa(Algo):
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
        
        ret = 0
        cur_gamma = 1
        end_t = min(t + 1, self.T_step + 1)
        for i in range(tgt_t + 1, end_t + 1):
            ret += cur_gamma * self.buffer.get(i)[2]
            cur_gamma *= self.gamma
        # print(ret)

        end_s, end_a = self.buffer.get(end_t)[:2]
        ret += cur_gamma * agent.q(end_s, end_a)

        tgt_s, tgt_a = self.buffer.get(tgt_t)[:2]
        terminate = tgt_t >= self.T_step

        ret = self.alpha * (ret - agent.q(tgt_s, tgt_a))
        # print(ret)
        return Command(ret, tgt_s, tgt_a, terminate)
