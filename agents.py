import random

import baseagent
import envs
import utils

from tilecoding import TileCoding
from network import Network


class Tabular(baseagent.Agent):
    def __init__(self, env: envs.DiscreteEnv):
        super().__init__(env, ['_q'])
        self._q = []
        if env is not None:
            self._q = utils.fit_shape(0, env.actions)

    def q(self, s, a):
        if s == self.env.T:
            return self.env.T_val
        return self._q[s][a]
    
    def update(self, diff, tgt_s, tgt_a):
        self._q[tgt_s][tgt_a] += diff


## ----------- BELOW NOT FINISHED -----------
class TileCode(baseagent.Agent):
    pass


class NN(baseagent.Agent):
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
