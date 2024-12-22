import agents
import approx
import envs


class Bruh(envs.DiscreteEnv):
    def next_state(self, s, a, *args, **kwargs):
        return super().next_state(s, a, *args, **kwargs)


agent = agents.Sarsa(approx.Tabular(), Bruh())
agent.load(file_name='envtest', env_name='envtest')
print(agent.env.actions)
print(agent.approx._q)