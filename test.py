import agents
import algos
import envs


class TestEnv(envs.DiscreteEnv):
    def next_state(self, s, a):
        if s == 0:
            return self.T, self.T_reward
        if self.actions[s][a] == 2:
            return 1, 1
        else:
            return 0, 1


# agent = agents.Tabular(algos.Sarsa(), TestEnv())
# agent.load(file_name='modeltest', env_name='envtest')
# print(agent.env.actions)
# print(agent._q)

agent = agents.Tabular(TestEnv([10,20], [[1],[2,3]], [1]))
agent.train(algos.Sarsa(alpha=0.1, gamma=0.9), n=100, batch_size=10)
# agent.train(algos.NStepSarsa(alpha=0.1, gamma=0.9, nstep=1), n=100, batch_size=10)
print(agent._q)
