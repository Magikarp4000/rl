from abc import ABC, abstractmethod
import random, time, datetime
import os
import json
import inspect
import threading

import numpy as np

from utils import (
    graph,
    get_dir,
    random_argmax,
    name,
    ReplayBuffer,
)
import data_transfer
from envs import Env
from params import Param, UniformDecay

from observer import Observable, Observer
from rlsignal import RLSignal


class ThreadInterrupt(Exception):
    pass


class Command:
    def __init__(self, tgt=None, s=None, a=None, terminate=False, update=True):
        self.tgt = tgt
        self.s = s
        self.a = a
        self.terminate = terminate
        self.update = update


class Agent(ABC, Observable, Observer):
    def __init__(self, env: Env, algo, extracter=None, config=[]):
        super().__init__()
        self.env = env
        self.algo = algo

        self.eps = None
        self.alpha = None

        self.ep = 0
        self.glo_steps = 0
        self.replay = ReplayBuffer()
        self.steps_list = []
        self._extracter = extracter
        self.thread = None
        self.running = False

        self._config = config
        self._metadata = {}
    
    def respond(self, obj, signal):
        if signal == RLSignal.TRAIN_CLICKED:
            self.running = True
            self.thread = threading.Thread(
                target=self.train,
                kwargs={'n': 1000, 'eps': UniformDecay(0.2, 0.05, decay=5*10**-5), 'alpha': Param(0.01),
                        'batch_size': 1, 'display_graph': False},
                daemon=True,
            )
            self.notify(RLSignal.TRAIN_START)
            self.thread.start()
        elif signal == RLSignal.TEST_CLICKED:
            self.running = True
            self.thread = threading.Thread(
                target=self.train,
                kwargs={'n': 1000, 'eps': Param(1), 'alpha': Param(0),
                        'batch_size': 1, 'display_graph': False},
                daemon=True,
            )
            self.thread.start()
        elif signal == RLSignal.STOP_SIMULATION:
            self.notify(RLSignal.TRAIN_END)
            self.running = False
            self.thread.join()

    def train(self, n, eps=Param(0.1), alpha=Param(0.1), maxstep=np.inf, expstart=False,
              batch_size=1, display_graph=True, save_params=True, save_time=True):
        self.eps = eps
        self.alpha = alpha
        
        if save_params:
            self._save_params(n, eps, expstart)
        
        start_time = time.time()
        try:
            self._train(n, maxstep, expstart, batch_size, display_graph)
        except ThreadInterrupt:
            return
        end_time = time.time()

        if save_time:
            self._config_meta({'duration': str(end_time - start_time)})
            self._config_meta({'time': str(datetime.datetime.now())})
    
    def test(self, num_steps=1, max_step=None, eps=0):
        return np.mean([self._single_test(max_step, eps)] * num_steps)
    
    def load(self, model_name, env_name):
        model_path, env_path = self._get_paths(model_name, env_name)

        self.env.load(env_path)
        print(f'Loaded environment {env_name}!')

        self._load_model(model_path)
        print(f'Loaded model {model_name}!')
    
    def save(self, model_name, env_name, overwrite_env=False):
        model_path, env_path = self._get_paths(model_name, env_name)
        
        if self._confirm_save_env(env_path, overwrite_env):
            self.env.save(env_path)
            print(f'Saved environment {env_name}!')

        if self._confirm_save_model(model_name, env_name, model_path):
            self._save_model(model_path)
            print(f'Saved model {model_name}!')

    def get_action(self, s, eps=0):
        if s == self.env.T:
            return self.env.T_action
        if random.random() < eps:
            return self.env.rand_action(s)
        else:
            return self.best_bhv_action(s)
    
    def best_bhv_action(self, s):
        return self.best_action(s)
    
    def bhv_action_vals(self, s):
        return self.action_vals(s)
    
    def best_bhv_action_val(self, s):
        return self.best_action_val(s)
    
    def action_vals(self, s):
        return [self.q(s, a) for a in self.env.action_spec(s)]

    def best_action(self, s, rand_tiebreak=True):
        if rand_tiebreak:
            return random_argmax(self.action_vals(s))
        else:
            return np.argmax(self.action_vals(s))
    
    def best_action_val(self, s):
        return np.max(self.action_vals(s))
    
    def action_prob(self, s, a):
        if s == self.env.T:
            return 1
        best_val = self.best_action_val(s)
        prob = self.eps / self.env.num_actions(s)
        if self.q(s, a) == best_val:
            num_best = self.action_vals(s).count(best_val)
            prob += (1 - self.eps) / num_best
        return prob

    def _config_meta(self, data):
        for val in data:
            self._metadata.update({val: data[val]})

    def _save_params(self, *args, **kwargs):
        try:
            train_args = inspect.getfullargspec(self.train)[0][2: 6]
            algo_args = self.algo.get_params()
            self._config_meta({'approximator': name(self)})
            self._config_meta({'algo': algo_args})
            self._config_meta({name: self._save_x(val) for name, val in zip(train_args, args)})
            self._config_meta(kwargs)
        except IndexError:
            pass
    
    def _save_x(self, x):
        try:
            return x.get_params()
        except AttributeError:
            return x
    
    def _train(self, n, maxstep, expstart, batch_size, display_graph):
        while self.ep < n:
            start_time = time.time()
            start_ep = self.ep
            while self.ep < min(n, start_ep + batch_size):
                self.notify(RLSignal.EP_START)
                steps = self._train_episode(maxstep, expstart)
                self.notify(RLSignal.EP_END)
                self.steps_list.append(steps)
                self.ep += 1
            end_time = time.time()
            if batch_size == 1:
                print(f"Episode {self.ep} complete in {round(end_time - start_time, 3)}s", end=", ")
            else:
                print(f"Episodes {start_ep + 1} - {self.ep} complete in {round(end_time - start_time, 3)}s", end=", ")
            print(f"{np.mean(self.steps_list[self.ep - batch_size:])} steps avg.")
        if display_graph:
            graph(self.steps_list, 'Episode', 'Num steps')

    def _train_episode(self, maxstep, expstart):
        s, a = self._init_state_action(expstart)
        self.algo.init_episode(s, a)

        cmd = Command()
        is_terminal = False
        steps = 0
        self.replay.create_new_ep(self.ep + 1)

        while not (cmd.terminate or steps > maxstep):
            if not self.running:
                self.replay.clear()
                self.glo_steps -= steps
                raise ThreadInterrupt()
            
            if not is_terminal:
                new_s, r = self.env.next_state(s, a)
                new_a = self.get_action(new_s, eps=self.eps())
                if new_s == self.env.T:
                    is_terminal = True
            
            cmd = self.algo(self, s, a, r, new_s, new_a, steps, is_terminal)
            data = self._extracter.extract(self, s, a, r, new_a, new_s, steps, cmd)
            self.replay.write(data)
            
            if cmd.update:
                self.update(cmd.tgt, cmd.s, cmd.a)
            self.eps.update(steps, self.glo_steps)
            self.alpha.update(steps, self.glo_steps)

            s, a = new_s, new_a
            steps += 1
            self.glo_steps += 1
        return steps
    
    def _init_state(self, expstart):
        if expstart:
            return self.env.rand_state()
        else:
            return self.env.rand_start_state()

    def _init_state_action(self, expstart=False):
        s = self._init_state(expstart)
        a = self.get_action(s, eps=1)  # Random action
        return s, a

    def _single_test(self, max_step=np.inf, eps=0):
        steps = 0
        s, a = self._init_state_action()
        while not (s == self.env.T or steps > max_step):
            a = self.get_action(s, eps=eps)
            s, _ = self.env.next_state(s, a)
            steps += 1
        return steps
    
    def _load_model(self, model_path):
        to_load = data_transfer.load(model_path)
        for name in to_load:
            if name == 'metadata':
                self._metadata = to_load[name]
            elif name in self._config:
                setattr(self, name, to_load[name])
        self.load_convert()
    
    def _save_model(self, model_path):
        self.save_convert()
        to_save = {'metadata': self._metadata}
        for name in self._config:
            try:
                value = self._save_x(getattr(self, name))
                json.dumps(value)
                to_save[name] = value
            except TypeError:
                print(f"SAVE_WARNING: '{name}' (type {type(value)}) is not JSON serializable!")
            except AttributeError:
                print(f"SAVE_WARNING: '{name}' doesn't exist!")
        data_transfer.save(model_path, to_save)
    
    def _confirm_save_env(self, env_path, overwrite_env):
        return overwrite_env or not os.path.exists(env_path)
    
    def _confirm_save_model(self, model_name, env_name, model_path):
        if os.path.exists(model_path):
            response = ""
            while response.lower() not in ("y", "n"):
                response = input(
                    f"Model {model_name} already exists for environment {env_name}. Overwrite (y/n)? ")
            return response.lower() == "y"
        return True

    def _get_paths(self, model_name, env_name):
        path = f"{get_dir()}/envs/{env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = f"{path}/{model_name}.json"
        env_path = f"{path}/env.json"
        return model_path, env_path

    def load_convert(self): pass
    def save_convert(self): pass
    def v(self, s): pass
    def v_prime(self, s): pass
    def q(self, s, a): pass
    def q_prime(self, s, a): pass
    @abstractmethod
    def update(self, tgt, s, a): pass
