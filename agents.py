import random

import numpy as np

from baseagent import Agent
from envs import DiscreteEnv
from utils import Buffer, fit_shape

from tilecoding import TileCoding
from network import Network


class Tabular(Agent):
    def __init__(self, env: DiscreteEnv, algo, alpha=0.1):
        super().__init__(env, algo, ['_q'])
        self.alpha = alpha
        
        self._q = None
        if env is not None:
            self._q = fit_shape(0, env.actions)

    def q(self, s, a):
        if s == self.env.T:
            return self.env.T_val
        return self._q[s][a]
    
    def update(self, tgt, s, a):
        self._q[s][a] += self.alpha * (tgt - self.q(s, a))


class TileCode(Agent):
    pass


class NN(Agent):
    def __init__(self, env, algo, nn: Network, batch):
        super().__init__(env, algo, ['nn'])
        self.nn = nn
        self.buffer = Buffer(batch)
    
    def q(self, s, a):
        if s == self.env.T:
            return self.env.T_val
        return self.action_vals(s).flat[a]
    
    def update(self, tgt, s, a):
        state = self.to_state(s)
        output = self.action_vals(s)
        output[a] = tgt
        self.buffer.update((state, output))
        if self.buffer.idx == 0 and self.buffer.get(1) is not None:
            self.nn.train(train_data=self.buffer(), epochs=10, mini_batch_size=self.buffer.size, eta=1)
    
    def to_state(self, s):
        return self.column(self.env.states[s])

    def column(self, vec):
        return np.reshape(vec, (len(vec), 1))
    
    def action_vals(self, s):
        if s == self.env.T:
            return [0]
        return self.nn.feedforward(self.to_state(s))
