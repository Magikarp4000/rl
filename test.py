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
        if self.actions[s][a] == 2:
            return 1, 1
        elif self.actions[s][a] == 3:
            if random.random() < 0.5:
                return 0, 1
            else:
                return 1, 1
        else:
            return 0, 1


# agent = agents.Tabular(algos.Sarsa(), TestEnv())
# agent.load(file_name='modeltest', env_name='envtest')
# print(agent.env.actions)
# print(agent._q)

agent = agents.Tabular(TestEnv())
agent.train(algos.NStepSarsa(alpha=0.1, gamma=1, nstep=5), n=1000, batch_size=10)
print(agent._q)
agent.save('v0.1b', 'testenv')
