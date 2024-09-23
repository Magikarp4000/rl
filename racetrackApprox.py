from agents import *
from racetrack_utils import *


class Approxtrack(Racetrack, Approximator):
    def __init__(self, track, start, finish, max_spe):
        super().__init__(track, start, finish, max_spe, False)
    

track, start, finish = get_track()
pro = Approxtrack(track, start, finish, 3)
pro.train('sarsa', 10, num_tiles=4, tile_frac=4, alpha=0.4)
