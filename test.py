import random

from agents import Tabular, NN
from algos import *
import envs
from network import Network


class TestEnv(envs.DiscreteEnv):
    def __init__(self):
        super().__init__(states=[[10],[20]], actions=[[1,5,6],[2,3,4]], start_states=[1])

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
# algo = PrioritizedSweep(TreeLearn(gamma=0.9, nstep=5), plan_algo=QLearn(), nsim=5)
algo = QLearn(gamma=0.9, nstep=1)
nn = Network([1, 10, 3], cost_type='mse')

agent = NN(env, algo, nn)
agent.train(n=1000, batch_size=10)

print([agent.nn.feedforward(agent.to_state(s)) for s in range(len(agent.env.states))])
# agent.save('vnn0.2b', 'testenv2')
