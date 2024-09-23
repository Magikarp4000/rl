import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def normalise_bounds(x, b):
    try:
        return (x - b[0]) / (b[1] - b[0])
    except:
        return x

def vectorise(speed, angle):
    dx = speed * np.sin(np.pi / 180 * angle)
    dy = speed * np.cos(np.pi / 180 * angle)
    return dx, dy
