import agents
import algos
import envs

import random


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


# agent = agents.Tabular(algos.Sarsa(), TestEnv())
# agent.load(file_name='modeltest', env_name='envtest')
# print(agent.env.actions)
# print(agent._q)

agent = agents.Tabular(TestEnv())
# agent.load('v0.1b', 'testenv')
agent.train(algos.NStepSarsa(alpha=0.1, gamma=1, nstep=5), n=1000, batch_size=10)
print(agent._q)
agent.save('v0.1b', 'testenv')
