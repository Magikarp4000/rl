import baseagent


class Sarsa(baseagent.Agent):
    def core(self, s, a, r, new_s, new_a, gamma=0.9, alpha=0.1):
        return alpha * (r + gamma * self.approx.q(new_s, new_a) - self.approx.q(s, a))
