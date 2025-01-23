from agents import *
from racetrack_utils import *


class TDtrack(Racetrack, TDAgent):
    pass


class Dynatrack(Racetrack, Dyna):
    pass


track, start, finish = get_track()
pro = Dynatrack(track, start, finish, 3)
pro.load('racetrack/vcircular4.3', load_actions=True)
# pro.train('qlearn', 20, n=20, alpha=0.1, eps=0.1, batch_size=1, rand_actions=True)
# pro.train('prioq+', 100000, n=10, gamma=1.0, alpha=0.1, eps=0.1, theta=0.1, batch_size=1000, rand_actions=True)
# pro.save('racetrack/vcircular4.3')
pro.animate(fps=20, policy='opt')