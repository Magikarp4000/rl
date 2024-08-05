import random, time, datetime
from abc import ABC, abstractmethod

import numpy as np
import data_transfer


class Agent(ABC):
    def __init__(self):
        self._data = []
        self._metadata = {}
        self.states = []
        self.actions = [[]]
        self.size = 0
        self.state_map = {}
    
    def core_init(self):
        self.set_state_actions()
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

    def load(self, file_name=None):
        to_load = data_transfer.load(file_name)
        for name in to_load:
            if name == 'metadata':
                self._metadata = to_load[name]
            elif name in self._data:
                setattr(self, name, to_load[name])
        self.load_convert()
        self.core_init()
        print(f'Loaded {file_name}.json!')
    
    def save(self, file_name=None):
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

    @abstractmethod
    def set_state_actions(self):
        pass


class DPAgent(Agent):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.config(['q', 'pi', 'q_record', 'pi_record'])
        self.q = [[]]
        self.pi = []
        self.q_record = []
        self.pi_record = []
        self.b = []
    
    def train(self, algo, num_epsd, gamma=1, eps=0.1, q=None, pi=None, batch_size=1, save_params=True, save_time=True):
        if save_params:
            self.config_meta({'algo': algo, 'num_epsd': num_epsd, 'gamma': gamma, 'eps': eps})
        if q:
            self.q = q
        else:
            self.q = [[0 for _ in self.actions[s]] for s in range(self.size)]
        if pi:
            self.pi = pi
        else:
            self.pi = [random_argmax(self.q[s]) for s in range(self.size)]
        self.q_record.clear()
        self.pi_record.clear()
        if algo == 'onpolicy':
            self.onpolicy(num_epsd, gamma, eps, batch_size)
        elif algo == 'offpolicy':
            self.offpolicy(num_epsd, gamma, batch_size)
        else:
            print("Invalid algorithm mate")
        if save_time:
            self.config_meta({'time': str(datetime.datetime.now())})
    
    def get_action(self, s):
        return np.random.choice(len(self.actions[s]), p=self.b[s])
    
    def on_epsd(self, gamma, eps, seq, nums):
        visited = set()
        g = 0
        for step in reversed(seq):
            s, a, reward = step
            g = reward + gamma * g
            temp = nums[s][a] if nums[s][a] else 1
            self.q[s][a] += 1 / temp * (g - self.q[s][a])
            nums[s][a] += 1
            visited.add(s)
        for s in visited:
            old_pi = self.pi[s]
            self.pi[s] = random_argmax(self.q[s])
            # self.b[s] = np.full(len(self.actions[s]), eps / len(self.actions[s]))
            # self.b[s][self.pi[s]] += 1 - eps
            self.b[s][old_pi] = eps / len(self.actions[s])
            self.b[s][self.pi[s]] = 1 - eps + eps / len(self.actions[s])
        return nums

    def onpolicy(self, num_epsd, gamma, eps, batch_size):
        self.b = [np.full(len(self.actions[s]), eps / len(self.actions[s]))
                  for s in range(self.size)]
        for s in range(self.size):
            self.b[s][self.pi[s]] += 1 - eps
        nums = [np.zeros(len(self.actions[s])) for s in range(self.size)]
        num = 0
        real_num = 0
        for num in range(num_epsd // batch_size):
            start_time = time.time()
            for _ in range(batch_size):
                real_num += 1
                seq = self.get_episode()
                # eps = max(0.1, min(1, 1 / np.sqrt(real_num)))
                nums = self.on_epsd(gamma, eps, seq, nums)
            end_time = time.time()
            print(f"Episodes {num * batch_size} - {(num + 1) * batch_size} complete in {round(end_time - start_time, 2)}s.")
            num += 1

    def off_epsd(self, gamma, seq, c):
        eps = 0.1
        visited = set()
        g = 0
        ratio = 1
        for step in reversed(seq):
            s, a, reward = step
            visited.add(s)
            g = reward + gamma * g
            c[s][a] += ratio
            self.q[s][a] += ratio / c[s][a] * (g - self.q[s][a])
            self.pi[s] = np.argmax(self.q[s])
            if a != self.pi[s]:
                for t in visited:
                    self.b[t] = np.full(len(self.actions[t]), eps / len(self.actions[t]))
                    self.b[t][self.pi[t]] += 1 - eps
                return c
            ratio *= 1 / self.b[s][a]
        for s in visited:
            self.b[s] = np.full(len(self.actions[s]), eps / len(self.actions[s]))
            self.b[s][self.pi[s]] += 1 - eps
        return c

    def offpolicy(self, num_epsd, gamma, batch_size):
        eps = 0.1
        self.b = [np.full(len(self.actions[s]), eps / len(self.actions[s]))
                  for s in range(self.size)]
        for s in range(self.size):
            self.b[s][self.pi[s]] += 1 - eps
        # self.b = [np.full(len(self.actions[s]), 1 / len(self.actions[s]))
        #           for s in range(self.size)]
        c = [[0 for _ in self.actions[s]] for s in range(self.size)]
        num = 0
        for num in range(num_epsd // batch_size):
            start_time = time.time()
            for _ in range(batch_size):
                seq = self.get_episode()
                c = self.off_epsd(gamma, seq, c)
            end_time = time.time()
            print(f"Episodes {num * batch_size} - {(num + 1) * batch_size} complete in {round(end_time - start_time, 2)}s.")
            num += 1
    
    def load_convert(self):
        pass
    
    def save_convert(self):
        pass

    def display(self, display_type):
        pass

    @abstractmethod
    def get_episode(self):
        pass


def random_argmax(arr):
    return int(np.random.choice(np.flatnonzero(arr == np.max(arr, axis=0))))
