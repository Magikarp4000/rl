import random, time, datetime, heapq
from abc import ABC, abstractmethod
import json

import numpy as np
from matplotlib import pyplot as pl

import data_transfer
from tilecoding import TileCoding


class Agent(ABC):
    def __init__(self, *args, **kwargs):
        # Constants
        self.terminal = 'T'
        self.terminal_reward = 0
        self.terminal_action = 0

        # Model data
        self._data = []
        self._metadata = {}

    def config(self, data):
        for val in reversed(data):
            if val not in self._data:
                self._data.insert(0, val)
    
    def config_meta(self, data):
        for val in data:
            self._metadata.update({val: data[val]})

    def load(self, file_name, load_actions=False):
        to_load = data_transfer.load(file_name)
        for name in to_load:
            if name == 'metadata':
                self._metadata = to_load[name]
            elif name in self._data:
                setattr(self, name, to_load[name])
        self.load_convert()
        self.core_init()
        if load_actions and 'actions' in to_load:
            self.actions = [[tuple(x) for x in state] for state in to_load['actions']]
        print(f'Loaded {file_name}.json!')
    
    def save(self, file_name):
        self.save_convert()
        to_save = {'metadata': self._metadata}
        for name in self._data:
            try:
                value = getattr(self, name)
                json.dumps(value)
                to_save[name] = value
            except TypeError:
                print(f"SAVE_WARNING: '{name}' (type {type(value)}) is not JSON serializable!")
            except AttributeError:
                print(f"SAVE_WARNING: '{name}' doesn't exist!")
            
        data_transfer.save(file_name, to_save)

    def load_convert(self):
        pass

    def save_convert(self):
        pass

    def _single_test(self, max_step, eps):
        steps = 0
        s, a = self.init_episode()
        while s != self.terminal and (max_step is None or steps < max_step):
            a = self.get_action(s, eps=eps)
            s, _ = self.next_state(s, a)
            steps += 1
        return steps

    def test(self, num_steps=1, max_step=None, eps=0):
        return np.mean([self._single_test(max_step, eps)] * num_steps)

    @abstractmethod
    def get_action(self, s, eps, *args, **kwargs):
        pass

    @abstractmethod
    def next_state(self, s, a, *args, **kwargs):
        pass

    @abstractmethod
    def init_episode(self, *args, **kwargs):
        pass


class TabularAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Environment & agent data
        self.states = []
        self.actions = [[]]
        self.size = 0
        self.state_map = {}
        self.starts = []

    def core_init(self, *args, **kwargs):
        self.set_state_actions(*args, **kwargs)
        self.size = len(self.states)
        self.state_map = {self.states[s]: s for s in range(self.size)}

    def state_to_index(self, state):
        try:
            return self.state_map[state]
        except TypeError:
            return self.state_map[tuple(state)]


class DPAgent(TabularAgent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config(['v', 'pi', 'v_record', 'pi_record'])
        self.v = np.array([])
        self.pi = np.array([])
        self.pi_record = []
        self.v_record = []

    def train(self, algo, gamma=0.9, theta=1, record=False, save_params=True, save_time=True):
        if save_params:
            self.config_meta({'algo': algo, 'gamma': gamma, 'theta': theta})
        self.size = len(self.states)
        self.v = np.array([0 for _ in range(self.size + 1)], dtype='float')
        self.pi = np.array([0 for _ in range(self.size + 1)], dtype='int')
        self.v_record.clear()
        self.pi_record.clear()
        if algo == 'policy':
            self.policy_iteration(gamma, theta, record)
        elif algo =='value':
            self.value_iteration(gamma, theta, record)
        else:
            print("Invalid algorithm mate")
        if save_time:
            self.config_meta({'time': str(datetime.datetime.now())})

    def policy_iteration(self, gamma, theta, record=False):
        num = 1
        stable = False
        while not stable:
            start_time = time.time()
            self.evaluate(gamma, theta, record)
            eval_time = time.time()
            stable = self.improve(gamma, record)
            end_time = time.time()
            self.display('values')
            self.display('policy')
            print(f"Policy evaluation completed in {round(eval_time - start_time, 2)}s.")
            print(f"Policy improvement completed in {round(end_time - eval_time, 2)}s.")
            print(f"Iteration {num} completed in {round(end_time - start_time, 2)}s.")
            num += 1
    
    def evaluate(self, gamma, theta, record):
        delta = theta + 1
        while delta >= theta:
            delta = 0
            for s in range(self.size):
                old_val = self.v[s]
                self.v[s] = self.eval_state_action(s, self.pi[s], gamma)
                delta = max(delta, abs(self.v[s] - old_val))
                # Animation
                if record:
                    self.v_record.append((s, self.v[s]))
    
    def eval_state_action(self, s, a, gamma):
        cur_val = 0
        probs, action_reward = self.get_prob(s, a)
        for new_data, prob in probs.items():
            new_s, reward = new_data
            reward += action_reward
            cur_val += prob * (reward + gamma * self.v[new_s])
        return cur_val
    
    def improve(self, gamma, record):
        stable = True
        for s in range(self.size):
            old_action = self.pi[s]
            action_vals = tuple(self.eval_state_action(s, a, gamma) 
                                for a in range(len(self.actions[s])))
            self.pi[s] = np.argmax(action_vals)
            if action_vals[old_action] != action_vals[self.pi[s]]:
                stable = False
            # Animation
            if record:
                self.pi_record.append((s, self.actions[s][self.pi[s]]))
        return stable
    
    def value_iteration(self, gamma, theta, record=False):
        delta = theta + 1
        num = 0
        while delta >= theta:
            start_time = time.time()
            delta = self.value_sweep(gamma, record)
            end_time = time.time()
            self.display('values')
            print(f"Iteration {num} completed in {round(end_time - start_time, 2)}s.")
            num += 1
        self.improve(gamma, record)

    def value_sweep(self, gamma, record):
        delta = 0
        for s in range(self.size):
            old_val = self.v[s]
            self.v[s] = max([self.eval_state_action(s, a, gamma)
                             for a in range(len(self.actions[s]))])
            delta = max(delta, abs(self.v[s] - old_val))
            # Animation
            if record:
                self.v_record.append((s, self.v[s]))
        return delta
    
    def load_convert(self):
        self.v = np.array(self.v)
        self.pi = np.array(self.pi)
    
    def save_convert(self):
        self.v = self.v.tolist()
        self.pi = self.pi.tolist()
    
    def display(self, display_type):
        pass

    def next_state(self, s, a):
        pass

    def init_episode(self, s, eps):
        pass

    @abstractmethod
    def get_prob(self, s, a):
        pass


class MCAgent(TabularAgent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config(['q', 'pi', 'b'])
        self.q = [[]]
        self.pi = []
        self.b = []
    
    def init_train(self, eps, rand_actions, q, pi):
        if rand_actions:
            for s in range(self.size):
                random.shuffle(self.actions[s])
            self.config(['actions'])
        self.q = q if q else [[0 for _ in self.actions[s]] for s in range(self.size)]
        self.pi = pi if pi else [int(np.argmax(self.q[s])) for s in range(self.size)]
        self.b = [np.full(len(self.actions[s]), eps / len(self.actions[s]))
                  for s in range(self.size)]
        for s in range(self.size):
            self.b[s][self.pi[s]] = 1 - eps + eps / len(self.actions[s])

    def train(self, algo, num_ep, gamma=1, eps=0.1, sampling_type='weighted', rand_actions=True, q=None, pi=None, batch_size=1, save_params=True, save_time=True):
        self.init_train(eps, rand_actions, q, pi)
        if save_params:
            self.config_meta({'algo': algo, 'num_ep': num_ep, 'gamma': gamma, 'eps': eps, 'sampling_type': sampling_type})
        if algo == 'onpolicy':
            self.on_policy(num_ep, gamma, eps, batch_size)
        elif algo == 'offpolicy':
            self.off_policy(num_ep, gamma, eps, sampling_type, batch_size)
        else:
            print("Invalid algorithm mate")
        if save_time:
            self.config_meta({'time': str(datetime.datetime.now())})
    
    def get_action(self, s):
        return np.random.choice(len(self.actions[s]), p=self.b[s])
    
    def on_policy(self, num_ep, gamma, eps, batch_size):
        nums = [np.zeros(len(self.actions[s])) for s in range(self.size)]
        ep = 0
        for _ in range(num_ep // batch_size):
            start_time = time.time()
            for _ in range(batch_size):
                seq = self.get_episode()
                nums = self.on_ep(gamma, eps, seq, nums)
                ep += 1
            end_time = time.time()
            print(f"Episodes {ep - batch_size} - {ep} complete in {round(end_time - start_time, 2)}s.")
    
    def on_ep(self, gamma, eps, seq, nums):
        visited = set()
        g = 0
        for step in reversed(seq):
            s, a, reward = step
            visited.add(s)
            g = reward + gamma * g
            temp = nums[s][a] if nums[s][a] else 1
            self.q[s][a] += 1 / temp * (g - self.q[s][a])
            nums[s][a] += 1
        for s in visited:
            self.pi[s] = random_argmax(self.q[s])
        self.update_b(eps, visited)
        return nums
    
    def off_policy(self, num_ep, gamma, eps, sampling_type, batch_size):
        c = [np.zeros(len(self.actions[s])) for s in range(self.size)]
        ep_num = 0
        for ep_num in range(num_ep // batch_size):
            start_time = time.time()
            for _ in range(batch_size):
                seq = self.get_episode()
                c = self.off_ep(gamma, eps, sampling_type, seq, c)
            end_time = time.time()
            print(f"Episodes {ep_num * batch_size} - {(ep_num + 1) * batch_size} complete in {round(end_time - start_time, 2)}s.")
            ep_num += 1
    
    def off_ep(self, gamma, eps, sampling_type, seq, c):
        visited = set()
        g = 0
        w = 1
        for step in reversed(seq):
            s, a, reward = step
            visited.add(s)
            g = reward + gamma * g
            c[s][a] += w
            self.q[s][a] += w / c[s][a] * (g - self.q[s][a])
            self.pi[s] = int(np.argmax(self.q[s]))
            if a != self.pi[s]:
                break
            w *= 1 / self.b[s][a]
        self.update_b(eps, visited)
        return c
    
    def update_b(self, eps, to_update=None):
        if to_update is None:
            to_update = [i for i in range(self.size)]
        for s in to_update:
            self.b[s] = np.full(len(self.actions[s]), eps / len(self.actions[s]))
            self.b[s][self.pi[s]] = 1 - eps + eps / len(self.actions[s])
    
    def load_convert(self):
        for s in range(self.size):
            try:
                self.b[s] = np.array(self.b[s])
            except:
                pass
    
    def save_convert(self):
        for s in range(self.size):
            try:
                self.b[s] = self.b[s].tolist()
            except:
                pass

    def display(self, display_type):
        pass

    @abstractmethod
    def get_episode(self):
        pass


class TDAgent(TabularAgent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config(['q', 'pi', 'b'])
        self.q = []
        self.pi = []
        self.b = []

    def init_train(self, eps, rand_actions, q=None, pi=None):
        if rand_actions:
            for s in range(self.size):
                random.shuffle(self.actions[s])
            self.config(['actions'])
        self.q = q if q else [[0 for _ in self.actions[s]] for s in range(self.size + 1)]
        self.q[self.size][0] = 0
        self.pi = pi if pi else [int(np.argmax(self.q[s])) for s in range(self.size + 1)]
        self.b = [[eps / len(self.actions[s]) for _ in self.actions[s]]
                  for s in range(self.size + 1)]
        for s in range(self.size + 1):
            self.b[s][self.pi[s]] = 1 - eps + eps / len(self.actions[s])
    
    def get_action(self, s, eps=None):
        if eps:
            if random.random() < eps:
                return random.randint(0, len(self.actions[s]) - 1)
            return self.pi[s]
        return np.random.choice(len(self.actions[s]), p=self.b[s])

    def update_b(self, eps, to_update=None, prev_opt=None):
        if to_update is None:
            to_update = [i for i in range(self.size)]
        if prev_opt is not None and isinstance(to_update, int):
            self.b[to_update][prev_opt] = eps / len(self.actions[to_update])
            self.b[to_update][self.pi[to_update]] = 1 - eps + eps / len(self.actions[to_update])
        else:
            for s in to_update:
                self.b[s] = [eps / len(self.actions[s]) for _ in self.actions[s]]
                self.b[s][self.pi[s]] = 1 - eps + eps / len(self.actions[s])

    def train(self, 
        algo, 
        num_ep, 
        n=1, 
        gamma=1, 
        alpha=0.1, 
        eps=0.1, 
        explore_starts=False, 
        rand_actions=False, 
        batch_size=1, 
        q=None, 
        pi=None, 
        save_params=True, 
        save_time=True,
    ):
        self.init_train(eps, rand_actions, q, pi)
        if save_params:
            self.config_meta({'algo': algo, 'num_ep': num_ep, 'n': n, 'gamma': gamma, 'alpha': alpha, 'eps': eps, 'explore_starts': explore_starts})
        ep = 0
        while ep < num_ep:
            start_time = time.time()
            for _ in range(batch_size):
                if algo in ['sarsa', 'offsarsa', 'expsarsa', 'expoffsarsa', 'tree']:
                    self.sarsa_ep(algo, n, gamma, alpha, eps, explore_starts)
                elif algo == 'qlearn':
                    self.qlearn_ep(gamma, alpha, eps, explore_starts)
                elif algo == 'dqlearn':
                    self.double_qlearn_ep(gamma, alpha, eps, explore_starts)
                ep += 1
            end_time = time.time()
            print(f"Episodes {ep - batch_size} - {ep} complete in {round(end_time - start_time, 2)}s.")
        if save_time:
            self.config_meta({'time': str(datetime.datetime.now())})

    def init_episode(self, eps=0, explore_starts=False):
        s = self.state_to_index(random.choice(self.starts))
        if explore_starts:
            s = random.randint(0, self.size - 1)
        a = self.get_action(s, eps)
        return s, a

    def sarsa_ep(self, algo, n, gamma, alpha, eps, explore_starts):
        s, a = self.init_episode(explore_starts=explore_starts)
        cache = [(0, 0, 0) for _ in range(n + 1)]   # (state, action, reward)
        T = np.inf
        t = 0
        while t < T + n - 1:
            if t < T:
                # get state, reward at step t+1
                new_s, reward = self.next_state(s, a)
                # check if self.terminal state
                if new_s >= self.size:
                    T = t + 1
                # get action at step t+1
                new_a = self.get_action(new_s, eps)
                # store state, action, reward at step t+1
                cache[(t + 1) % (n + 1)] = (new_s, new_a, reward)
            # update step t-n+1 with return from steps t-n+1:t+1
            prev = t - n + 1
            if prev >= 0:
                ret = 0
                if algo == 'tree':
                    ret = self.get_tree_return(algo, gamma, prev, t, T, cache)
                else:
                    ret = self.get_sarsa_return(algo, gamma, prev, t, T, cache)
                rho = self.get_importance_ratio(algo, prev, t, T, cache)
                # update step t-n+1
                prev_s, prev_a = cache[prev % (n + 1)][:2]
                self._single_q_update(prev_s, prev_a, ret, alpha, rho)
                prev_opt = self.pi[prev_s]
                self.pi[prev_s] = int(np.argmax(self.q[prev_s]))
                self.update_b(eps, prev_s, prev_opt)
            s, a = new_s, new_a
            t += 1
    
    def get_tree_return(self, algo, gamma, prev, t, T, cache):
        ret = cache[min(t + 1, T) % len(cache)][2]
        if t + 1 < T:
            end_s = cache[(t + 1) % len(cache)][0]
            ret += gamma * self.q[end_s][self.pi[end_s]]
        for i in range(min(t, T - 1), prev, -1):
            cur_s, cur_a, reward = cache[i % len(cache)]
            if cur_a == self.pi[cur_s]:
                ret = reward + gamma * ret
            else:
                ret = reward + gamma * self.q[cur_s][self.pi[cur_s]]
        return ret

    def get_sarsa_return(self, algo, gamma, prev, t, T, cache):
        ret = 0
        cur_gamma = 1
        # prev+1 to min(t, T)
        for i in range(prev + 1, min(t + 1, T + 1)):
            ret += cur_gamma * cache[i % len(cache)][2] # reward
            cur_gamma *= gamma
        # t+1
        if t + 1 < T:
            end_s, end_a = cache[(t + 1) % len(cache)][:2]
            if algo == 'expsarsa':
                ret += cur_gamma * sum(prob * val for prob, val in zip(self.b[end_s], self.q[end_s]))
            elif algo == 'expoffsarsa':
                ret += cur_gamma * self.q[end_s][self.pi[end_s]]
            elif algo == 'sarsa' or algo == 'offsarsa':
                ret += cur_gamma * self.q[end_s][end_a]
        return ret

    def get_importance_ratio(self, algo, prev, t, T, cache):
        if algo != 'offsarsa' and algo != 'expoffsarsa':
            return 1
        rho = 1
        # offsarsa: prev+1 to min(t, T-1)
        # expoffsarsa: prev+1 to min(t-1, T-1)
        _end = min(t, T) if algo == 'expoffsarsa' else min(t + 1, T)
        for i in range(prev + 1, _end):
            cur_s, cur_a = cache[i % len(cache)][:2] # state, action
            if cur_a == self.pi[cur_s]:
                rho *= 1 / self.b[cur_s][cur_a]
            else:
                return 0
        return rho

    def qlearn_ep(self, gamma, alpha, eps, explore_starts):
        s, a = self.init_episode(explore_starts=explore_starts)
        visited = set()
        while s < self.size:
            visited.add(s)
            new_s, reward = self.next_state(s, a)
            new_a = self.get_action(new_s, eps)
            ret = reward + gamma * max(self.q[new_s])
            self._single_q_update(s, a, ret, alpha)
            self.pi[s] = random_argmax(self.q[s])
            s, a = new_s, new_a
        self.update_b(eps, visited)

    def double_qlearn_ep(self, gamma, alpha, eps, explore_starts):
        s, a = self.init_episode(explore_starts=explore_starts)
        visited = set()
        while s < self.size:
            visited.add(s)
            new_s, reward = self.next_state(s, a)
            new_a = self.get_action(new_s, eps)
            ret1 = reward + gamma * self.q2[new_s][random_argmax(self.q1[s])]
            ret2 = reward + gamma * self.q1[new_s][random_argmax(self.q2[s])]
            self._double_q_update(self, s, a, ret1, ret2, alpha)
            self.pi[s] = random_argmax(self.q[s])
            s, a = new_s, new_a
        self.update_b(eps, visited)
    
    def _single_q_update(self, s, a, ret, alpha, rho=1):
        self.q[s][a] += alpha * rho * (ret - self.q[s][a])

    def _double_q_update(self, s, a, ret1, ret2, alpha, rho=1):
        if random.randint(0, 1) == 0:
            self.q1[s][a] += alpha * rho * (ret1 - self.q1[s][a])
        else:
            self.q2[s][a] += alpha * rho * (ret2 - self.q2[s][a])
        self.q[s][a] = self.q1[s][a] + self.q2[s][a]
    
    def display(self, display_type):
        pass


class Dyna(TabularAgent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Base variables
        self.algos = ['q', 'q+', 'prioq', 'prioq+']

        # Model data
        self.config(['q', 'pi'])
        self.q = []
        self.pi = []
        self.model = {}
        self.b = []

        # Q+
        self.last_visit = []

        # Prioritised sweeping
        self.reverse_model = {}
        self.priors = {}
        self.pq = []
        heapq.heapify(self.pq)

        # Training parameters
        self.algo = None
        self.num_ep = None
        self.gamma = None
        self.alpha = None
        self.eps = 0
        self.kappa = None
        self.theta = None
        self.n = None
        self.explore_starts = False
    
    def init_train(self, rand_actions, save_params, q=None, pi=None):
        # Save parameters
        if save_params:
            self.config_meta({
                'algo': self.algo, 
                'num_ep': self.num_ep, 
                'n': self.n, 
                'gamma': self.gamma, 
                'alpha': self.alpha, 
                'eps': self.eps, 
                'kappa': self.kappa, 
                'theta': self.theta,
                'explore_starts': self.explore_starts
            })
        
        # Randomise actions
        if rand_actions:
            for s in range(self.size):
                random.shuffle(self.actions[s])
            self.config(['actions'])
        
        # Initialise model data
        self.q = q if q is not None else [[0 for _ in self.actions[s]] for s in range(self.size + 1)]
        self.q[self.size][0] = 0
        self.pi = pi if pi is not None else [int(np.argmax(self.q[s])) for s in range(self.size + 1)]
        self.b = [[self.eps / len(self.actions[s]) for _ in self.actions[s]]
                  for s in range(self.size + 1)]
        for s in range(self.size + 1):
            self.b[s][self.pi[s]] = 1 - self.eps + self.eps / len(self.actions[s])
    
    def get_action(self, s, eps=None):
        if s == self.terminal:
            return self.terminal_action
        if eps is None:
            if self.eps is not None:
                eps = self.eps
        if eps:
            if random.random() < eps:
                return random.randint(0, len(self.actions[s]) - 1)
            return self.pi[s]
        return np.random.choice(len(self.actions[s]), p=self.b[s])

    def init_episode(self, eps=0):
        s = self.state_to_index(random.choice(self.starts))
        if self.explore_starts:
            s = random.randint(0, self.size - 1)
        a = self.get_action(s, eps)
        if self.algo == 'q+':
            self.last_visit = [[0 for _ in self.actions[s]] for s in range(self.size + 1)]
        return s, a

    def train(self, 
        algo, 
        num_ep,
        n=1,
        gamma=1.0,
        alpha=0.1,
        eps=0.1,
        kappa=0.05,
        theta=0.05,
        explore_starts=False,
        rand_actions=True,
        batch_size=1,
        q=None,
        pi=None,
        save_params=True,
        save_time=True,
    ):
        if algo not in self.algos:
            print("Invalid algorithm mate")
        else:
            self.algo = algo
            self.num_ep = num_ep
            self.gamma = gamma
            self.alpha = alpha
            self.eps = eps
            self.kappa = kappa
            self.theta = theta
            self.n = n
            self.explore_starts = explore_starts
            self.init_train(rand_actions, save_params, q, pi)
            ep = 0
            while ep < num_ep:
                start_time = time.time()
                for _ in range(batch_size):
                    self.qlearn_ep()
                    ep += 1
                end_time = time.time()
                print(f"Episodes {ep - batch_size} - {ep} complete in {round(end_time - start_time, 2)}s.")
            if save_time:
                self.config_meta({'time': str(datetime.datetime.now())})
    
    def qlearn_ep(self):
        t = 0
        s, a = self.init_episode()
        while s != self.terminal:
            new_s, new_a, reward = self.qlearn_step(s, a, t)
            self.update_model(s, a, new_s, reward)
            self.sim_exp()
            s, a = new_s, new_a
            t += 1

    def qlearn_step(self, s, a, t):
        new_s, reward = self.next_state(s, a)
        new_a = self.get_action(new_s)

        if self.algo == 'q+' or self.algo == 'prioq+':
            reward += self.kappa * np.sqrt(t - self.last_visit[s][a])
            self.last_visit[s][a] = t
        
        prior = self._single_q_update(s, a, new_s, reward)

        if self.algo == 'prioq' or self.algo == 'prioq+':
            self._prio_update(prior, s, a)
        
        self.pi[s] = int(np.argmax(self.q[s]))

        return new_s, new_a, reward

    def update_model(self, s, a, new_s, reward):
        if self.algo == 'prioq' or self.algo == 'prioq+':
            if (s, a) in self.model:
                old_new_s = self.model[(s, a)][0]
                self.reverse_model[old_new_s].remove((s, a))
            if new_s not in self.reverse_model:
                self.reverse_model[new_s] = [(s, a)]
            else:
                self.reverse_model[new_s].append((s, a)) 
        self.model[(s, a)] = (new_s, reward)
    
    def sim_exp(self):
        if self.algo == 'prioq' or self.algo == 'prioq+':
            for _ in range(self.n):
                if not self.pq:
                    break
                _, s, a = heapq.heappop(self.pq)
                new_s, reward = self.model[(s, a)]
                self._single_q_update(s, a, new_s, reward)
                self.pi[s] = int(np.argmax(self.q[s]))
                for prev_s, prev_a in self.reverse_model[s]:
                    _, prev_reward = self.next_state(prev_s, prev_a)
                    prior = prev_reward + self.gamma * max(self.q[s]) - self.q[prev_s][prev_a]
                    self._prio_update(prior, prev_s, prev_a)
        else:
            samp = random.sample(list(self.model.keys()), min(self.n, len(self.model)))
            for s, a in samp:
                new_s, reward = self.model[(s, a)]
                self._single_q_update(s, a, new_s, reward)
                self.pi[s] = int(np.argmax(self.q[s]))
    
    def _single_q_update(self, s, a, new_s, reward):
        if s == self.terminal or new_s == self.terminal:
            return self.terminal_reward
        _update = reward + self.gamma * max(self.q[new_s]) - self.q[s][a]
        self.q[s][a] += self.alpha * _update
        return _update

    def _prio_update(self, prior, s, a):
        threshold = self.theta if (s, a) not in self.priors else max(self.theta, self.priors[(s, a)])
        if prior >= threshold:
            heapq.heappush(self.pq, (-prior, s, a))


class Approximator(Agent):
    def __init__(self, base_actions=[], bounds=[], start_bounds=[], dim=0, *args, **kwargs):
        super().__init__()
        # Base variables
        self.algos = ['td', 'qlearn', 'sarsa']

        # Environment & agent data
        self.config(['base_actions', 'bounds', 'start_bounds'])
        self.base_actions = base_actions
        self.bounds = bounds
        self.start_bounds = start_bounds

        # Model data
        self.config(['dim', 'd', 'w', 'z'])
        self.dim = dim
        self.d = None
        self.w = []

        # Eligibility trace
        self.z = []

        # Training parameters
        self.algo = None
        self.num_ep = None
        self.num_steps = None
        self.gamma = None
        self.alpha = None
        self.beta = None
        self.eps = 0
        self.a_eps = None
        self.lamd = None
        self.explore_starts = False

        # Tile coding
        self.config(['num_layers', 'num_per_dim', 'offsets', 'mod_list'])
        self.num_layers = None
        self.num_per_dim = None
        self.offsets = None
        self.mod_list = None
        
        self.converter = None

    def init_train(self, save_params):
        # Save parameters
        if save_params:
            self.config_meta({
                'algo': self.algo,
                'num_ep': self.num_ep,
                'num_steps': self.num_steps,
                'gamma': self.gamma,
                'alpha': self.alpha,
                'beta': self.beta,
                'eps': self.eps,
                'a_eps': self.a_eps,
                'lambda': self.lamd,
                'explore_starts': self.explore_starts,
            })
        
        # Tile coding
        self.converter = TileCoding(self.num_layers, self.dim, self.bounds, self.num_per_dim, 
                                    self.offsets, self.mod_list)
        
        # Initialise model data
        self.d = self.converter.total * self.num_layers
        self.w = np.zeros(self.d)

        # Eligibility trace
        if self.lamd is not None:
            self.z = np.zeros(self.d)

    def init_state(self):
        s = [np.random.uniform(bound[0], bound[1]) for bound in self.start_bounds]
        return s

    def init_episode(self, eps=0):
        s = self.init_state()
        a = self.get_action(s, eps)
        return s, a

    def train(self,
        algo,
        num_ep=1,
        num_steps=None,
        gamma=1.0,
        alpha=0.1,
        beta=0.1,
        eps=0.1,
        a_eps=0.1,
        lamd=None,
        explore_starts=False,
        num_layers=8,
        num_per_dim=[],
        offsets=[],
        mod_list=[],
        batch_size=1,
        save_params=True,
        save_time=True,
    ):
        if algo not in self.algos:
            print("Invalid algorithm mate!\nList of valid algorithms:")
            for x in self.algos:
                print(f'- {x}')
            return
        self.algo = algo
        self.num_ep = num_ep
        self.num_steps = num_steps
        self.gamma = gamma
        self.alpha = alpha / num_layers
        self.beta = beta
        self.eps = eps
        self.a_eps = a_eps
        self.lamd = lamd
        self.explore_starts = explore_starts
        self.num_layers = num_layers
        self.num_per_dim = num_per_dim
        self.offsets = offsets
        self.mod_list = mod_list
        self.init_train(save_params)

        start_time = time.time()
        if algo == 'td':
            self.td()
        elif algo == 'lstd':
            self.lstd()
        elif algo == 'sarsa':
            if num_steps is None:
                self.sarsa(batch_size)
            else:
                self.sarsa_cont()
        end_time = time.time()
        if save_time:
            self.config_meta({'duration': str(end_time - start_time)})
            self.config_meta({'time': str(datetime.datetime.now())})
    
    def get_action(self, s, eps=None):
        if s == self.terminal:
            return self.terminal_action
        if eps is None:
            eps = self.eps
        if np.random.random() < eps:
            return np.random.randint(len(self.base_actions))
        else:
            tmp = [self.q(s, a) for a in range(len(self.base_actions))]
            return random_argmax(tmp)

    def sarsa(self, batch_size):
        ep = 0
        steps_list = []
        while ep < self.num_ep:
            start_time = time.time()
            for _ in range(batch_size):
                steps = self.sarsa_ep()
                steps_list.append(steps)
                ep += 1
            end_time = time.time()
            print(f"Episodes {ep - batch_size} - {ep} complete in {round(end_time - start_time, 2)}s.")
        graph(steps_list, 'Num steps', 'Episode')
    
    def sarsa_ep(self):
        s, a = self.init_episode()
        steps = 0
        while s != self.terminal:
            new_s, reward = self.next_state(s, a)
            new_a = self.get_action(new_s)

            gradient = self.q_prime(s, a)
            if self.lamd is not None:
                self.z = self.gamma * self.lamd * self.z + self.q_prime(s, a)
                gradient = self.z
            
            self.w += self.alpha * (reward + self.q(new_s, new_a) - self.q(s, a)) * gradient

            s, a = new_s, new_a
            steps += 1
        return steps
    
    def sarsa_cont(self):
        s, a = self.init_episode()
        avg_reward = 0
        step = 0
        while step < self.num_steps:
            new_s, reward = self.next_state(s, a)
            new_a = self.get_action(new_s)
            
            error = reward - avg_reward + self.q(new_s, new_a) - self.q(s, a)
            avg_reward += self.beta * error

            self.w += self.alpha * error * self.q_prime(s, a)

            s, a = new_s, new_a
            step += 1

    def td(self):
        ep = 0
        while ep < self.num_ep:
            self.td_ep()
            ep += 1

    def td_ep(self):
        s, a = self.init_episode()
        while s != self.terminal:
            a = self.pi[s]
            new_s, reward = self.next_state(s, a)
            self.w += self.alpha * (reward + self.gamma * self.v(new_s) - self.v(s)) * self.v_prime(s)
    
    def lstd(self):
        a_inv = (1 / self.a_eps) * np.identity(self.d)
        b = np.zeros(self.d).transpose()
        ep = 0
        while ep < self.num_ep:
            a_inv, b = self.lstd_ep(a_inv, b)
            ep += 1

    def lstd_ep(self, a_inv, b):
        s, a = self.init_episode()
        x = self.x[s]
        while s != self.terminal:
            a = self.pi[s]
            new_s, reward = self.next_state(s, a)
            new_x = self.x[new_s]
            v = (a_inv.transpose() * (x - self.gamma * new_x)).transpose()
            a_inv -= (a_inv * x * v) / (1 + v * x) # sherman-morrison
            b += reward * x
            self.w = a_inv * b
            s, x = new_s, new_x
        return a_inv, b

    def v(self, s: list):
        if s == self.terminal:
            return 0
        res = 0
        tiles = self.converter.encode(s)
        for tile in tiles:
            res += self.w[tile]
        return res

    def v_prime(self, s: list):
        if s == self.terminal:
            return np.zeros(self.d)
        x = np.zeros(self.d)
        tiles = self.converter.encode(s)
        for tile in tiles:
            x[tile] = 1
        return x
    
    def q(self, s: list, a: int):
        if s == self.terminal:
            return 0
        new_s, reward = self.next_state(s, a)
        return reward + self.v(new_s)

    def q_prime(self, s: list, a: int):
        if s == self.terminal:
            return np.zeros(self.d)
        new_s, _ = self.next_state(s, a)
        return self.v_prime(new_s)

    def save_convert(self):
        super().save_convert()
        self.w = self.w.tolist()
        self.z = self.z.tolist()
    
    def load_convert(self):
        super().load_convert()
        self.w = np.array(self.w)
        self.z = np.array(self.z)
        self.converter = TileCoding(self.num_layers, self.dim, self.bounds, self.num_per_dim, self.offsets)


def random_argmax(arr):
    return int(np.random.choice(np.flatnonzero(arr == np.max(arr, axis=0))))

def graph(y, xlabel=None, ylabel=None):
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.plot(np.arange(len(y)), y)
    pl.show()
