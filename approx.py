from abc import ABC, abstractmethod
import random

import envs
import utils

from tilecoding import TileCoding
from network import Network


class Approximator(ABC):
    def __init__(self, config=[]):
        self.config = config

    def init_train(self): pass
    def v(self, s): pass
    def v_prime(self, s): pass
    def q(self, s, a): pass
    def q_prime(self, s, a): pass
    @abstractmethod
    def update(self, s, a, r, new_s, new_a, diff): pass


class Tabular(Approximator):
    def __init__(self, env: envs.DiscreteEnv=None):
        super().__init__(['_q'])
        self._q = []
        if env is not None:
            self._q = utils.shape(env.actions, 0)

    def q(self, s, a):
        return self._q[s][a]
    
    def update(self, s, a, r, new_s, new_a, diff):
        self._q[s][a] += diff


## ----------- BELOW NOT FINISHED -----------
class TileCode(Approximator):
    pass


class NN(Approximator):
    def __init__(self, base_actions=[], bounds=[], start_bounds=[], dim=0):
        super().__init__(base_actions, bounds, start_bounds, dim)
        self.cache = []
        self.cache_idx = None
        self.capacity = None
        self.sample_size = None
        self.net_params = None
        self.network = None

    def get_data(self):
        """
        self.cache: (s, a, r, s', a')
        """
        minibatch = random.sample(self.cache, self.sample_size)
        X = [t[:2] for t in minibatch]
        y = [t[2] + self.gamma * self.q(t[3], t[4]) for t in minibatch]
        return list(zip(X, y))
    
    def approximate(self, s, a, r, new_s, new_a, *args, **kwargs):
        # Update memory for network
        self.cache_idx = (self.cache_idx + 1) % self.capacity
        self.cache[self.cache_idx] = (s, a, r, new_s, new_a)
        
        # Train network
        data = self.get_data()
        self.network.train(data, epochs=1, mini_batch_size=1, eta=0.1)
## ------------------------------------------
