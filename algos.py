from abc import ABC, abstractmethod
import inspect
import random
import heapq

import numpy as np

from baseagent import Agent, Command
from utils import Buffer, name


class Algo(ABC):
    """Base class for algorithms."""

    def get_params(self):
        args = inspect.getfullargspec(self.__init__)[0][1:]
        params = {'name': name(self)} | {arg: self._get_val(arg) for arg in args}
        return params
    
    def _get_val(self, arg):
        val = getattr(self, arg)
        if isinstance(val, Algo):
            return val.get_params()
        return val

    @abstractmethod
    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal): pass
    def init_episode(self, s, a): pass


class NStepAlgo(Algo):
    """
    Base class for N-step algorithms.

    Parameters
    ----------
    gamma : float [0, 1]
        Discount-rate.
    nstep : int [1, inf), default=5
        Number of steps used in bootstrapped updates.
    """
    def __init__(self, gamma=0.9, nstep=1):
        super().__init__()
        self.gamma = gamma
        self.nstep = nstep

        self.buffer = None
        self.T_step = None
    
    def init_episode(self, s, a):
        self.buffer = Buffer(self.nstep + 1, (None, None, None))  # (s, a, r)
        self.buffer.set(0, (s, a, 0))
        self.T_step = np.inf

    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal):
        if self.T_step is np.inf and is_terminal:
            self.T_step = t + 1

        if t < self.T_step:
            self.buffer.set(t + 1, (new_s, new_a, r))
        
        tgt_t = t - self.nstep + 1
        if tgt_t < 0:
            return Command(update=False)

        tgt_s, tgt_a = self.buffer.get(tgt_t)[:2]
        ret = self.get_return(agent, t, tgt_t)
        terminate = tgt_t >= self.T_step - 1

        return Command(ret, tgt_s, tgt_a, terminate)

    def get_return(self, agent: Agent, t, tgt_t):
        end_t = min(t + 1, self.T_step)
        ret = self.end_return(agent, *self.buffer.get(end_t))
        for i in range(end_t - 1, tgt_t, -1):
            ret = self.step_return(agent, *self.buffer.get(i), ret)
        return ret
    
    @abstractmethod
    def end_return(self, agent, s, a, r): pass
    @abstractmethod
    def step_return(self, agent, s, a, r, ret): pass


class Dyna(Algo):
    """
    Parameters
    ----------
    algo : Algo
        Algorithm used for direct RL.
    plan_algo : Algo
        Algorithm used for planning.
    nsim: int [1, inf)
        Number of steps per planning simulation.
    """
    def __init__(self, algo: Algo, plan_algo: Algo, nsim=1):
        super().__init__()
        self.algo = algo
        self.plan_algo = plan_algo
        self.nsim = nsim
        self.model = {}

    def init_episode(self, s, a):
        self.algo.init_episode(s, a)
        self.plan_algo.init_episode(s, a)

    def __call__(self, agent, s, a, r, new_s, new_a, t, is_terminal):
        if not is_terminal:
            self.model[(s, a)] = (new_s, r)
            self.simulate(agent)
        return self.algo(agent, s, a, r, new_s, new_a, t, is_terminal)

    def simulate(self, agent):
        samp = random.choices(list(self.model.keys()), k=self.nsim)
        for s, a in samp:
            new_s, r = self.model[(s, a)]
            new_a = agent.get_action(new_s)
            ret = self.plan_algo.end_return(agent, new_s, new_a, r)
            agent.update(ret, s, a)


class PrioritizedSweeping(Dyna):
    def __init__(self, algo, plan_algo, nsim=1, theta=0.05):
        super().__init__(algo, plan_algo, nsim)
        self.theta = theta

        self.pq = None
    
    def init_episode(self, s, a):
        super().init_episode(s, a)
        self.pq = []
        heapq.heapify(self.pq)


class ExploreBonus(Algo):
    """
    Parameters
    ----------
    algo : Algo
        Core algorithm.
    kappa : float [0, 1]
        Exploration bonus.
    """
    def __init__(self, algo: Algo, kappa=0.05):
        super().__init__()
        self.algo = algo
        self.kappa = kappa
        
        self._last_visit = None

    def init_episode(self, s, a):
        self.algo.init_episode(s, a)
        self._last_visit = {}
    
    def __call__(self, agent, s, a, r, new_s, new_a, t, is_terminal):
        if not is_terminal:
            r += self.kappa * np.sqrt(t - self.last_visit(s, a))
            self._last_visit[(s, a)] = t
        return self.algo(agent, s, a, r, new_s, new_a, t, is_terminal)

    def last_visit(self, s, a):
        if (s, a) not in self._last_visit:
            return 0
        return self._last_visit[(s, a)]


class Sarsa(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * agent.q(s, a)
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class ExpectedSarsa(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * sum([agent.action_prob(s, cur_a) * agent.q(s, cur_a)
                                     for cur_a in agent.env.action_spec(s)])
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class QLearn(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * agent.best_action_val(s)
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class TreeLearn(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * agent.best_action_val(s)
    
    def step_return(self, agent, s, a, r, ret):
        best_a = agent.best_action(s)
        tmp = ret if a == best_a else agent.q(s, best_a)
        return r + self.gamma * tmp


class OnPolicyTreeLearn(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * agent.best_action_val(s)
    
    def step_return(self, agent, s, a, r, ret):
        tmp = sum([agent.action_prob(s, cur_a) * (ret if cur_a == a else agent.q(s, cur_a))
                   for cur_a in agent.env.action_spec(s)])
        return r + self.gamma * tmp
