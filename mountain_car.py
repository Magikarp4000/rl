from agents import *
import numpy as np


class Car(Approximator):
    def __init__(self, base_actions=[-1, 0, 1], bounds=[(-1, 1), (-1, 1)], start_bounds=[(-1, 1), (0, 0)]):
        super().__init__(base_actions, bounds, start_bounds)
    
    def next_state(self, s, a):
        action = self.base_actions[a]
        pos, vel = s
        new_vel = np.clip(vel + 0.001 * action - 0.0025 * np.cos(3 * pos), self.bounds[1][0], self.bounds[1][1])
        new_pos = np.clip(pos + new_vel, self.bounds[0][0], self.bounds[0][1])
        if new_pos == self.bounds[0][0]:
            new_vel = 0
        new_s = [new_pos, new_vel]
        if new_pos == self.bounds[0][1]:
            new_s = -1
        reward = -1
        return new_s, reward

    def set_state_actions(self):
        return


car = Car(bounds=[(-1.2, 0.5), (-0.07, 0.07)], start_bounds=[(-0.6, -0.4), (0, 0)])
car.load('mountain/v1.0')
# car.train('sarsa', 500, gamma=1.0, alpha=0.4, eps=0.1, num_tiles=8, tile_frac=8, batch_size=10)
# car.save('mountain/v1.0')
print(car.test(100))
# car.train('sarsa', 0, num_tiles=8, tile_frac=8)
# print(car.get_tile_coding([-0.9879, -0.0525]))
