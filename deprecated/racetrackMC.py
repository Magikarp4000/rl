from agents import *
from racetrack_utils import *


class MCtrack(Racetrack, MCAgent):
    def __init__(self, track, start, finish, max_spe):
        super().__init__(track, start, finish, max_spe)
    
    def get_episode(self, fast_terminate=False, no_reset=False):
        seq = []
        start_x, start_y = random.choice(self.start)
        state = (start_x, start_y, 0, 0)
        while state[:2] not in self.finish_set:
            x, y, xspe, yspe = state
            s = self.state_to_index(state)
            a = self.get_action(s)
            dx, dy = self.actions[s][a]
            xspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
            yspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
            oldx, oldy = x, y
            x += xspe
            y += yspe
            if (x, y) not in self.track_set:
                if fast_terminate:
                    seq.append((s, a, -1000))
                    break
                elif no_reset:
                    x, y, xspe, yspe = oldx, oldy, xspe, yspe
                else:
                    x, y, xspe, yspe = self.reset()
            state = (x, y, xspe, yspe)
            reward = -1
            seq.append((s, a, reward))
        return seq


track, start, finish = get_track()
pro = MCtrack(track, start, finish, 3)
pro.load('racetrack/vD2.0')
pro.train('onpolicy', 100000, eps=0.1, batch_size=1)
# pro.save('racetrack/vE3.0')
pro.animate(15, policy='opt')