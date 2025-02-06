from baseagent import Agent, Command


class Extracter:
    def extract(self, agent: Agent, s, a, r, new_a, new_s, t, cmd: Command):
        data = {}
        data['s'] = s
        data['a'] = a
        data['r'] = r
        data['new_a'] = new_a
        data['new_s'] = new_s
        data['t'] = t
        data['cmd'] = cmd
        return data


class ExtracterPlus(Extracter):
    def extract(self, agent, s, a, r, new_a, new_s, t, cmd):
        data = super().extract(agent, s, a, r, new_a, new_s, t, cmd)
        data['avals'] = agent.bhv_action_vals(s).flat
        data['action'] = agent.env.actions[a]
        return data
