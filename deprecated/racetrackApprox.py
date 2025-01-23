from agents import *
from racetrack_utils import *


class Approxtrack(Racetrack, Approximator):
    def __init__(self, track, start, finish, max_spe):
        super().__init__(track, start, finish, max_spe, False)
    

track, start, finish = get_track()
pro = Approxtrack(track, start, finish, 3)
pro.train('sarsa', 10, alpha=0.4, num_layers=4, num_per_dim=4, 
          offsets=[1,3,5,7,9], mod_list=[False,True,False,False,False])
