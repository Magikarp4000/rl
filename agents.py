import random, time, datetime, heapq
from abc import ABC, abstractmethod
import json
import inspect

import numpy as np
from matplotlib import pyplot as pl

import data_transfer
import envs
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


class Agent(ABC):
    def __init__(self, approx: Approximator, env: envs.Env):
        super().__init__()
        self.approx = approx
        self.env = env
        
        self.terminal = 'T'
        self.terminal_reward = 0
        self.terminal_action = 0

        self._metadata = {}
    
    def _config_meta(self, data):
        for val in data:
            self._metadata.update({val: data[val]})

    def _save_params(self, *args, **kwargs):
        try:
            train_args = inspect.getargs(self.train)[1: 4]
            core_args = inspect.getargs(self.core)[6:]
            self._config_meta({name: val for name, val in zip(train_args, args)})
            self._config_meta({name: val for name, val in zip(core_args, args)})
            self._config_meta(kwargs)
        except IndexError:
            pass
    
    def _train(self, n, eps, expstart, batch_size, *args, **kwargs):
        ep = 0
        steps_list = []
        self.cache_idx = 0
        while ep < n:
            start_time = time.time()
            for _ in range(batch_size):
                steps = self._train_episode(eps, expstart, *args, **kwargs)
                steps_list.append(steps)
                ep += 1
            end_time = time.time()
            print(f"Episodes {ep - batch_size} - {ep} complete in {round(end_time - start_time, 2)}s.")
        graph(steps_list, 'Num steps', 'Episode')

    def _train_episode(self, eps, expstart, *args, **kwargs):
        s, a = self._init_state_action(expstart)
        steps = 0
        while s != self.terminal:
            new_s, r = self.env.next_state(s, a)
            new_a = self._get_action(new_s, eps=eps)
            diff = self.core(s, a, r, new_s, new_a, *args, **kwargs)
            self.approx.update(s, a, r, new_s, new_a, diff)
            s, a = new_s, new_a
            steps += 1
        return steps
    
    def _init_state(self, expstart):
        if expstart:
            return self.env.rand_state()
        else:
            return self.env.rand_start_state()

    def _init_state_action(self):
        s = self._init_state()
        a = self._get_action(s)
        return s, a

    def _get_action(self, s, eps=0):
        if random.random() < eps:
            return self.env.rand_action(s)
        else:
            return np.argmax([self.approx.q(s, a) for a in range(len(self.env.base_actions[s]))])
    
    def _single_test(self, max_step, eps):
        steps = 0
        s, a = self._init_state_action()
        while s != self.terminal and (max_step is None or steps < max_step):
            a = self._get_action(s, eps=eps)
            s, _ = self.env.next_state(s, a)
            steps += 1
        return steps

    def load(self, file_name, env_name, load_actions=False):
        self.env.load(env_name)
        print(f'Loaded environment {file_name}.json!')

        to_load = data_transfer.load(file_name)
        for name in to_load:
            if name == 'metadata':
                self._metadata = to_load[name]
            elif name in self.approx.config:
                setattr(self.approx, name, to_load[name])
        self.load_convert()
        if load_actions and 'actions' in to_load:
            self.actions = [[tuple(x) for x in state] for state in to_load['actions']]
        print(f'Loaded model {file_name}.json!')
    
    def save(self, file_name):
        self.save_convert()
        to_save = {'metadata': self._metadata}
        for name in self.approx.get_data():
            try:
                value = getattr(self, name)
                json.dumps(value)
                to_save[name] = value
            except TypeError:
                print(f"SAVE_WARNING: '{name}' (type {type(value)}) is not JSON serializable!")
            except AttributeError:
                print(f"SAVE_WARNING: '{name}' doesn't exist!")
        data_transfer.save(file_name, to_save)
    
    def train(self, n, eps=0.1, expstart=False, batch_size=1, save_params=True, save_time=True, *args, **kwargs):
        if save_params:
            self._save_params(n, eps, expstart, *args, **kwargs)
        
        self.approx.init_train(*args, **kwargs)

        start_time = time.time()
        self._train(n, eps, batch_size, *args, **kwargs)
        end_time = time.time()

        if save_time:
            self.config_meta({'duration': str(end_time - start_time)})
            self.config_meta({'time': str(datetime.datetime.now())})
    
    def test(self, num_steps=1, max_step=None, eps=0):
        return np.mean([self._single_test(max_step, eps)] * num_steps)

    @abstractmethod
    def core(self, s, a, r, new_s, new_a, *args, **kwargs): pass
    def load_convert(self): pass
    def save_convert(self): pass


class Sarsa(Agent):
    def core(self, s, a, r, new_s, new_a, gamma=0.9, alpha=0.1):
        return alpha * (r + gamma * self.approx.q(new_s, new_a) - self.approx.q(s, a))


class Tabular(Approximator):
    def __init__(self, env: envs.DiscreteEnv=None):
        super().__init__(['_q'])
        self._q = []
        if env is not None:
            self._q = shape(env.actions, 0)

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

def random_argmax(arr):
    return int(np.random.choice(np.flatnonzero(arr == np.max(arr, axis=0))))

def graph(y, xlabel=None, ylabel=None):
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.plot(np.arange(len(y)), y)
    pl.show()

def shape(val, arr):
    if not isinstance(arr, list):
        return val
    res = []
    for x in arr:
        res.append(shape(val, x))
    return res


# ----- TEST -----
class Bruh(envs.DiscreteEnv):
    def next_state(self, s, a, *args, **kwargs):
        return super().next_state(s, a, *args, **kwargs)


agent = Sarsa(Tabular(), Bruh())
agent.load(file_name='envtest', env_name='envtest')
print(agent.env.actions)
print(agent.approx._q)
