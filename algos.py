from abc import ABC, abstractmethod
import inspect
import random

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


class Sarsa(Algo):
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


class QLearn(Algo):
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


class NStepAlgo(Algo):
    """
    Base class for N-step algorithms.

    Parameters
    ----------
    alpha : float [0, 1]
        Step-size.
    gamma : float [0, 1]
        Discount-rate.
    nstep : int [1, inf), default=5
        Number of steps used in bootstrapped updates.
    """
    def __init__(self, alpha=0.1, gamma=0.9, nstep=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.nstep = nstep

        self.buffer = None
        self.T_step = None
    
    def init_episode(self, s, a):
        self.buffer = Buffer(self.nstep + 1, (0, 0, 0))
        self.buffer.set(0, (s, a, 0))
        self.T_step = np.inf

    def __call__(self, agent: Agent, s, a, r, new_s, new_a, t, is_terminal):
        if self.T_step is np.inf and is_terminal:
            self.T_step = t

        if not is_terminal:
            self.buffer.set(t + 1, (new_s, new_a, r))
        
        tgt_t = t - self.nstep + 1
        if tgt_t < 0:
            return Command(no_update=True)

        tgt_s, tgt_a = self.buffer.get(tgt_t)[:2]
        ret = self.alpha * (self.get_return(agent, t, tgt_t) - agent.q(tgt_s, tgt_a))
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


class NStepSarsa(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * agent.q(s, a)
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class NStepExpectedSarsa(NStepAlgo):
    def end_return(self, agent, s, a, r):
        return r + self.gamma * sum([agent.action_prob(s, cur_a) * agent.q(s, cur_a)
                                     for cur_a in agent.env.action_spec(s)])
    
    def step_return(self, agent, s, a, r, ret):
        return r + self.gamma * ret


class NStepQLearn(NStepAlgo):
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
        ret = self.algo(agent, s, a, r, new_s, new_a, t, is_terminal)
        if not is_terminal:
            self.update(s, a, r, new_s, new_a)
            self.simulate(agent)
        return ret
    
    def update(self, s, a, r, new_s, new_a):
        self.model[(s, a)] = (new_s, r)

    def simulate(self, agent):
        samp = random.choices(list(self.model.keys()), k=self.nsim)
        for s, a in samp:
            new_s, r = self.model[(s, a)]
            new_a = agent.get_action(new_s)
            self.step_simulate(agent, s, a, r, new_s, new_a)
    
    def step_simulate(self, agent, s, a, r, new_s, new_a):
        cmd = self.plan_algo(agent, s, a, r, new_s, new_a, t=0, is_terminal=False)
        agent.update(cmd.diff, cmd.tgt_s, cmd.tgt_a)


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