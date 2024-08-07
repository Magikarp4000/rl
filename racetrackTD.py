from agents import *
from racetrack_utils import *


class TDtrack(Racetrack, TDAgent):
    def __init__(self, track, start, finish, max_spe):
        super().__init__(track, start, finish, max_spe)
    
    def next_state(self, s, a):
        x, y, xspe, yspe = self.states[s]
        dx, dy = self.actions[s][a]
        nxspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
        nyspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
        nx = x + nxspe
        ny = y + nyspe
        reward = -1
        if (nx, ny) in self.finish_set:
            return self.size, reward
        if (nx, ny) not in self.track_set:
            nx, ny, nxspe, nyspe = self.reset()
        new_s = self.state_to_index((nx, ny, nxspe, nyspe))
        return new_s, reward


track, start, finish = get_track()
pro = TDtrack(track, start, finish, 3)
# pro.load('racetrack/vfinalboss', load_actions=True)
# pro.train(500000, alpha=0.1, eps=0.1, batch_size=1000)
pro.save('racetrack/vcircular')
pro.animate(12, policy='opt')