import random

import agents
from algos import *
import envs


class TestEnv(envs.DiscreteEnv):
    def __init__(self):
        super().__init__([10,20], [[1],[2,3,4]], [1])

    def next_state(self, s, a):
        if s == 0:
            return self.T, self.T_reward
        r = 1
        new_s = 0
        if self.actions[s][a] == 2:
            new_s = 1
        elif self.actions[s][a] == 3:
            if random.random() < 0.5:
                new_s = 0
            else:
                new_s = 1
        return new_s, r


agent = agents.Tabular(TestEnv())
algo = ExploreBonus(TreeLearn(nstep=5))
agent.train(algo, n=1000, batch_size=10)
print(agent._q)
# agent.save('v0.2b', 'testenv')
