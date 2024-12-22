import random, time, datetime
from abc import ABC, abstractmethod
import json
import inspect

import numpy as np

import utils
import data_transfer
import envs


class Agent(ABC):
    def __init__(self, algo, env: envs.Env, config=[]):
        super().__init__()
        self.algo = algo
        self.env = env

        self._config = config
        self._metadata = {}
    
    def _config_meta(self, data):
        for val in data:
            self._metadata.update({val: data[val]})

    def _save_params(self, *args, **kwargs):
        try:
            train_args = inspect.getfullargspec(self.train)[0][1: 4]
            algo_args = self.algo.getargs()
            self._config_meta({name: val for name, val in zip(train_args, args)})
            self._config_meta({name: val for name, val in zip(algo_args, args)})
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
        utils.graph(steps_list, 'Num steps', 'Episode')

    def _train_episode(self, eps, expstart, *args, **kwargs):
        s, a = self._init_state_action(expstart)
        steps = 0
        while s != self.env.T:
            new_s, r = self.env.next_state(s, a)
            new_a = self._get_action(new_s, eps=eps)
            diff = self.algo(self, s, a, r, new_s, new_a, *args, **kwargs)
            self.update(s, a, r, new_s, new_a, diff)
            s, a = new_s, new_a
            steps += 1
        return steps
    
    def _init_state(self, expstart):
        if expstart:
            return self.env.rand_state()
        else:
            return self.env.rand_start_state()

    def _init_state_action(self, expstart=False):
        s = self._init_state(expstart)
        a = self._get_action(s)
        return s, a

    def _get_action(self, s, eps=0):
        if s == self.env.T:
            return self.env.T_action
        if random.random() < eps:
            return self.env.rand_action(s)
        else:
            return np.argmax([self.q(s, a) for a in self.env.action_spec(s)])
    
    def _single_test(self, max_step, eps):
        steps = 0
        s, a = self._init_state_action()
        while s != self.env.T and (max_step is None or steps < max_step):
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
            elif name in self._config:
                setattr(self, name, to_load[name])
        self.load_convert()
        if load_actions and 'actions' in to_load:
            self.actions = [[tuple(x) for x in state] for state in to_load['actions']]
        print(f'Loaded model {file_name}.json!')
    
    def save(self, file_name, env_name):
        self.env.save(env_name)
        print(f'Saved environment to {file_name}.json!')

        self.save_convert()
        to_save = {'metadata': self._metadata}
        for name in self._config:
            try:
                value = getattr(self, name)
                json.dumps(value)
                to_save[name] = value
            except TypeError:
                print(f"SAVE_WARNING: '{name}' (type {type(value)}) is not JSON serializable!")
            except AttributeError:
                print(f"SAVE_WARNING: '{name}' doesn't exist!")
        data_transfer.save(file_name, to_save)
        print(f'Saved model to {file_name}.json!')
    
    def train(self, n, eps=0.1, expstart=False, batch_size=1, save_params=True, save_time=True, *args, **kwargs):
        if save_params:
            self._save_params(n, eps, expstart, *args, **kwargs)
        
        self.init_train(*args, **kwargs)

        start_time = time.time()
        self._train(n, eps, expstart, batch_size, *args, **kwargs)
        end_time = time.time()

        if save_time:
            self._config_meta({'duration': str(end_time - start_time)})
            self._config_meta({'time': str(datetime.datetime.now())})
    
    def test(self, num_steps=1, max_step=None, eps=0):
        return np.mean([self._single_test(max_step, eps)] * num_steps)

    def load_convert(self): pass
    def save_convert(self): pass
    def init_train(self): pass
    def v(self, s): pass
    def v_prime(self, s): pass
    def q(self, s, a): pass
    def q_prime(self, s, a): pass
    @abstractmethod
    def update(self, s, a, r, new_s, new_a, diff): pass
