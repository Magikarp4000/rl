import random, time, datetime
from abc import ABC, abstractmethod

import numpy as np
import data_transfer


class Agent(ABC):
    def __init__(self, *args, **kwargs):
        self._data = []
        self._metadata = {}
        self.states = []
        self.actions = [[]]
        self.size = 0
        self.state_map = {}
    
    def core_init(self, *args, **kwargs):
        self.set_state_actions(*args, **kwargs)
        self.size = len(self.states)
        self.state_map = {self.states[s]: s for s in range(self.size)}

    def state_to_index(self, state):
        try:
            return self.state_map[state]
        except TypeError:
            return self.state_map[tuple(state)]

    def config(self, data):
        for val in reversed(data):
            if val not in self._data:
                self._data.insert(0, val)
    
    def config_meta(self, data):
        for val in reversed(data):
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
        for val in self._data:
            try:
                to_save[val] = getattr(self, val)
            except AttributeError:
                print(f"SAVE_ERROR: Attribute '{val}' doesn't exist!")
        data_transfer.save(file_name, to_save)

    def load_convert(self):
        pass

    def save_convert(self):
        pass

    def get_action(self, s):
        pass

    @abstractmethod
    def set_state_actions(self):
        pass


class DPAgent(Agent):
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

    @abstractmethod
    def get_prob(self, s, a):
        pass


class MCAgent(Agent):
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


class TDAgent(Agent):
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

    def init_ep(self, explore_starts=False):
        s = self.state_to_index(random.choice(self.starts))
        if explore_starts:
            s = random.randint(0, self.size - 1)
        a = self.get_action(s)
        return s, a

    def sarsa_ep(self, algo, n, gamma, alpha, eps, explore_starts):
        s, a = self.init_ep(explore_starts)
        cache = [(0, 0, 0) for _ in range(n + 1)]   # (state, action, reward)
        T = np.inf
        t = 0
        while t < T + n - 1:
            if t < T:
                # get state, reward at step t+1
                new_s, reward = self.next_state(s, a)
                # check if terminal state
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
        s, a = self.init_ep(explore_starts)
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
        s, a = self.init_ep(explore_starts)
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

    @abstractmethod
    def next_state(self):
        pass


class Dyna(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config(['q', 'pi', 'b'])
        self.q = []
        self.pi = []
        self.model = {}
        self.b = []
        self.algo = None
        self.num_ep = None
        self.gamma = None
        self.alpha = None
        self.eps = None
        self.kappa = None
        self.n = None
        self.explore_starts = False
        self.last_visit = []
    
    def init_train(self, rand_actions, save_params, q=None, pi=None):
        if save_params:
            self.config_meta({
                'algo': self.algo, 
                'num_ep': self.num_ep, 
                'n': self.n, 
                'gamma': self.gamma, 
                'alpha': self.alpha, 
                'eps': self.eps, 
                'kappa': self.kappa, 
                'explore_starts': self.explore_starts
            })
        if self.algo == 'q+':
            self.last_visit = [[0 for _ in self.actions[s]] for s in range(self.size + 1)]
        if rand_actions:
            for s in range(self.size):
                random.shuffle(self.actions[s])
            self.config(['actions'])
        self.q = q if q is not None else [[0 for _ in self.actions[s]] for s in range(self.size + 1)]
        self.q[self.size][0] = 0
        self.pi = pi if pi is not None else [int(np.argmax(self.q[s])) for s in range(self.size + 1)]
        self.b = [[self.eps / len(self.actions[s]) for _ in self.actions[s]]
                  for s in range(self.size + 1)]
        for s in range(self.size + 1):
            self.b[s][self.pi[s]] = 1 - self.eps + self.eps / len(self.actions[s])
    
    def get_action(self, s, eps=None):
        if eps is None:
            if self.eps is not None:
                eps = self.eps
        if eps:
            if random.random() < eps:
                return random.randint(0, len(self.actions[s]) - 1)
            return self.pi[s]
        return np.random.choice(len(self.actions[s]), p=self.b[s])

    def init_ep(self):
        s = self.state_to_index(random.choice(self.starts))
        if self.explore_starts:
            s = random.randint(0, self.size - 1)
        a = self.get_action(s)
        return s, a

    def train(self, 
        algo, 
        num_ep,
        n=1,
        gamma=1.0,
        alpha=0.1,
        eps=0.1,
        kappa=0.05,
        explore_starts=False,
        rand_actions=True,
        batch_size=1,
        q=None,
        pi=None,
        save_params=True,
        save_time=True,
    ):
        self.algo = algo
        self.num_ep = num_ep
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.kappa = kappa
        self.n = n
        self.explore_starts = explore_starts
        self.init_train(rand_actions, save_params, q, pi)
        ep = 0
        t = 0
        while ep < num_ep:
            start_time = time.time()
            for _ in range(batch_size):
                t = self.qlearn_ep(t)
                ep += 1
            end_time = time.time()
            print(f"Episodes {ep - batch_size} - {ep} complete in {round(end_time - start_time, 2)}s.")
        if save_time:
            self.config_meta({'time': str(datetime.datetime.now())})
    
    def qlearn_ep(self, t):
        s, a = self.init_ep()
        while s < self.size:
            new_s, new_a, reward = self.qlearn_step(s, a, t)
            self.model[(s, a)] = (new_s, reward)
            self.sim_exp()
            s, a = new_s, new_a
            t += 1
        return t

    def qlearn_step(self, s, a, t):
        new_s, reward = self.next_state(s, a)
        new_a = self.get_action(new_s)
        if self.algo == 'q+':
            reward += self.kappa * np.sqrt(t - self.last_visit[s][a])
            # print(s, self.actions[s][a], t-self.last_visit[s][a])
            self.last_visit[s][a] = t
        self._single_q_update(s, a, new_s, reward)
        self.pi[s] = int(np.argmax(self.q[s]))
        return new_s, new_a, reward
    
    def sim_exp(self):
        if self.model:
            for _ in range(self.n):
                s, a = random.choice(list(self.model.keys()))
                new_s, reward = self.model[(s, a)]
                self._single_q_update(s, a, new_s, reward)
                self.pi[s] = int(np.argmax(self.q[s]))
    
    def _single_q_update(self, s, a, new_s, reward):
        self.q[s][a] += self.alpha * (reward + self.gamma * max(self.q[new_s]) - self.q[s][a])


    @abstractmethod
    def next_state(self, s, a):
        pass


def random_argmax(arr):
    return int(np.random.choice(np.flatnonzero(arr == np.max(arr, axis=0))))
