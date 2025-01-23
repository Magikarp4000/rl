import random
from enum import Enum

from agents import Tabular, NN
from algos import *
from envs import DiscreteEnv, ContinuousEnv
from network import NetParams
from params import Param, UniformDecay, SampAvg


class TestEnv(DiscreteEnv):
    def __init__(self):
        super().__init__(states=[[0], [1]], actions=[[1,5,6],[4,3,2]], start_states=[1])

    def next_state(self, s, a):
        new_s, r = self.T, self.T_reward
        if self.actions[s][a] == 2:
            new_s, r = 1, 1
        elif self.actions[s][a] == 3:
            if random.random() < 0.5:
                new_s, r = 1, 1
        return new_s, r


class Cell(Enum):
    EMPTY = 0
    AGENT = 1
    ITEM = 2


class LBF(ContinuousEnv):
    def __init__(self, w, h, num_items):
        actions = ['U', 'D', 'L', 'R', 'C', 'X']
        b_x = [[0, w] for _ in range(num_items + 1)] # x-coord (+agent)
        b_y = [[0, h] for _ in range(num_items + 1)] # y-coord (+agent)
        b_exist = [[0, 1] for _ in range(num_items)] # item exists
        sb_exist = [[1, 1] for _ in range(num_items)]
        bounds = b_x + b_y + b_exist
        start_bounds = b_x + b_y + sb_exist
        super().__init__(actions, bounds, start_bounds)
        
        self.w = w
        self.h = h
        self.num_items = num_items
        self.move_map = {
            'U': [0, -1],
            'D': [0, 1],
            'L': [-1, 0],
            'R': [1, 0],
            'X': [0, 0]
        }
    
    def next_state(self, s, a, *args, **kwargs):
        agent_x, agent_y = s[0], s[self.num_items + 1]
        items_x = s[1: self.num_items + 1]
        items_y = s[self.num_items + 2: -(self.num_items)]
        item_exists = s[-self.num_items:]

        if not any(item_exists):
            return self.T, self.T_reward

        new_item_exists = s[-self.num_items:]
        reward = 0
        if self.actions[a] == 'C':
            for x, y in self.move_map.values():
                for i, (i_x, i_y) in enumerate(zip(items_x, items_y)):
                    if (agent_x + x, agent_y + y) == (i_x, i_y) and item_exists[i]:
                        reward += 1
                        new_item_exists[i] = 0
        else:
            agent_x = np.clip(agent_x + self.move_map[self.actions[a]][0], 0, self.w - 1)
            agent_y = np.clip(agent_y + self.move_map[self.actions[a]][1], 0, self.h - 1)
        new_s = [agent_x] + [agent_y] + items_x + items_y + new_item_exists
        return new_s, reward

    def rand_start_state(self):
        return [random.randint(*bound) for bound in self.start_bounds]
    
    def rand_state(self):
        return [random.randint(*bound) for bound in self.bounds]


env = LBF(3,4,1)
# algo = PrioritizedSweep(TreeLearn(gamma=0.9, nstep=5), plan_algo=QLearn(), nsim=5)
algo = QLearn(gamma=0.9, nstep=2)
nn = NetParams([6], cost_type='mse')

agent = NN(env, algo, nn, batch=20, upd_interval=100)
agent.train(n=1000, eps=Param(0.1), alpha=Param(0.1),
            batch_size=10, display_graph=True)

# print([agent.action_vals(s) for s in range(len(agent.env.states))])
print(agent.tnn.weights)
print(agent.tnn.biases)
agent.save('v0.1b', 'lbf')
