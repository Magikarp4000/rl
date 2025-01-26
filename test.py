import random

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


# env = LBF(10,6,1)
# # algo = PrioritizedSweep(TreeLearn(gamma=0.9, nstep=5), plan_algo=QLearn(), nsim=5)
# algo = QLearn(gamma=0.9, nstep=2)
# nn = NetParams([6], cost_type='mse')

# agent = NN(env, algo, nn, batch=20, upd_interval=100)
# agent.train(n=1000, eps=Param(0.1), alpha=Param(0.1),
#             batch_size=10, display_graph=True)

# print([agent.action_vals(s) for s in range(len(agent.env.states))])

# agent = NN(LBF(3, 4, 1), QLearn(gamma=0.9, nstep=2))
# agent.load('v0.1b', 'lbf')
# print(agent.bnn['weights'])

# print(agent.tnn.weights)
# print(agent.tnn.biases)
# agent.save('v0.1b', 'lbf2')
