import agents
import algos
import envs


class Bruh(envs.DiscreteEnv):
    def next_state(self, s, a, abc):
        return super().next_state(s, a, abc)


agent = agents.Tabular(algos.Sarsa(), Bruh())
agent.load(file_name='modeltest', env_name='envtest')
print(agent.env.actions)
print(agent._q)