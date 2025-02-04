import random

import numpy as np

from baseagent import Agent
from envs import Env, DiscreteEnv
from utils import VariableBuffer, fit_shape, random_argmax

from tilecoding import TileCoding
from network import NetParams, Network


class Tabular(Agent):
    def __init__(self, env: DiscreteEnv, algo):
        super().__init__(env, algo, ['_q'])
        
        self._q = None
        if env is not None:
            self._q = fit_shape(0, env.actions)

    def q(self, s, a):
        if s == self.env.T:
            return self.env.T_val
        return self._q[s][a]
    
    def update(self, tgt, s, a):
        self._q[s][a] += self.alpha() * (tgt - self.q(s, a))


class TileCode(Agent):
    pass


class NN(Agent):
    def __init__(self, env: Env, algo, netparams: NetParams=None,
                 batch=20, upd_interval=100, buf_size=1000):
        super().__init__(env, algo, ['batch', 'upd_interval', 'buf_size', 'bnn', 'tnn'])
        if netparams is not None:
            netparams.set_io(env.state_size(), env.num_actions(0))
            self.bnn = Network(netparams)
            self.tnn = Network(netparams)
        self.batch = batch
        self.upd_interval = upd_interval
        self.buf_size = buf_size
        self.buffer = VariableBuffer(self.buf_size)
    
    def load(self):
        self.buffer = VariableBuffer(self.buf_size)
    
    def update(self, tgt, s, a):
        state = self.to_state(s)
        output = self.action_vals_xnn(s, self.bnn)
        output[a] = tgt
        # print(output)
        self.buffer.update((state, output))
        self.bnn.train(train_data=random.sample(self.buffer(), min(self.buffer.cur_size(), self.batch)),
                       epochs=1,
                       mini_batch_size=self.batch,
                       eta=self.alpha())
        if self.upd_cycle():
            self.tnn.update(self.bnn)

    def upd_cycle(self):
        return (self.glo_steps != 0 and self.glo_steps % self.upd_interval == 0)

    def to_state(self, s):
        if isinstance(self.env, DiscreteEnv):
            return self.column(self.env.states[s])
        return self.column(s)

    def column(self, vec):
        return np.reshape(vec, (len(vec), 1))
    
    def q(self, s, a):
        if s == self.env.T:
            return self.env.T_val
        return self.action_vals(s).flat[a]
    
    def action_vals(self, s):
        return self.action_vals_xnn(s, self.tnn)
    
    def best_bhv_action(self, s):
        return random_argmax(self.action_vals_xnn(s, self.bnn))
    
    def action_vals_xnn(self, s, xnn: Network):
        if s == self.env.T:
            return [0]
        return xnn.feedforward(self.to_state(s))
