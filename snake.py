import numpy as np

from envs import ContinuousEnv


class Snake(ContinuousEnv):
    def __init__(self, w, h):
        actions = ['U', 'D', 'L', 'R']
        bounds = [[0, w - 1], [0, h - 1], [0, w - 1], [0, h - 1]]
        super().__init__(actions, bounds, bounds)
        self.speed = 0.1
        self.move_map = {
            'U': [0, -1],
            'D': [0, 1],
            'L': [-1, 0],
            'R': [1, 0],
        }
    
    def next_state(self, s, a, *args, **kwargs):
        agent_x, agent_y, item_x, item_y = self.decode_state(s)
        action = self.actions[a]
        raw_x = agent_x + self.speed * self.move_map[action][0]
        raw_y = agent_y + self.speed * self.move_map[action][1]
        new_x = np.clip(raw_x, *self.bounds[0])
        new_y = np.clip(raw_y, *self.bounds[1])

        raw_s = [raw_x, raw_y, item_x, item_y]
        new_s = [new_x, new_y, item_x, item_y]
        dist_reward = 0.5 if (self.manhat_dist(raw_s) - self.manhat_dist(s) < 0) else -0.5
        reward = -1 + dist_reward
        # print(dist_reward, end=" ", flush=True)
        # print(self.actions[a], dist_reward)
        if self.manhat_dist(new_s) < 1:
            return self.T, 1
        return new_s, reward
    
    def dist(self, s):
        agent_x, agent_y, item_x, item_y = self.decode_state(s)
        return np.sqrt((agent_x - item_x) ** 2 + (agent_y - item_y) ** 2)

    def manhat_dist(self, s):
        agent_x, agent_y, item_x, item_y = self.decode_state(s)
        return abs(agent_x - item_x) + abs(agent_y - item_y)

    def decode_state(self, s):
        return s