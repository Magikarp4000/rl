import agents
import algos
import envs


class TestEnv(envs.DiscreteEnv):
    def next_state(self, s, a):
        if s == 0:
            return self.T, self.T_reward
        if self.actions[a] == 2:
            return 1, 1
        else:
            return 0, 1


# agent = agents.Tabular(algos.Sarsa(), TestEnv())
# agent.load(file_name='modeltest', env_name='envtest')
# print(agent.env.actions)
# print(agent._q)

agent = agents.Tabular(algos.Sarsa(), TestEnv([10,20], [[1],[2,3]], [1]))
agent.train(n=1000, save_params=False)
print(agent._q)