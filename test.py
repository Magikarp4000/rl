import random

from agents import Tabular
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


env = TestEnv()
algo = Dyna(ExploreBonus(TreeLearn(gamma=0.9, nstep=5)), plan_algo=QLearn(), nsim=0)
# algo = TreeLearn(gamma=0.9, nstep=1)

agent = Tabular(env, algo, alpha=0.1)
agent.train(n=1000, batch_size=10)

print(agent._q)
# agent.save('v0.4b', 'testenv')
