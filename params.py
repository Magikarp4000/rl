import numpy as np


class Param:
    def __init__(self, x, amin=-np.inf, amax=np.inf):
        self.x = x
        self.amin = amin
        self.amax = amax
    
    def __call__(self):
        return self.x

    def update(self, t, glo_t): pass


class SampAvg(Param):
    def update(self, t, glo_t):
        self.x = np.clip(1 / (glo_t + 1), self.amin, self.amax)


class UniformDecay(Param):
    def __init__(self, x, amin=-np.inf, amax=np.inf, decay=0):
        super().__init__(x, amin, amax)
        self.decay = decay
    
    def update(self, t, glo_t):
        self.x = np.clip(self.x - self.decay, self.amin, self.amax)
