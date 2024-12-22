import os
import numpy as np
import matplotlib.pyplot as pl


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def random_argmax(arr):
    return int(np.random.choice(np.flatnonzero(arr == np.max(arr, axis=0))))

def normalise_bounds(x, b):
    try:
        return (x - b[0]) / (b[1] - b[0])
    except:
        return x

def vectorise(magnitude, angle):
    dx = magnitude * np.cos(np.pi / 180 * angle)
    dy = magnitude * np.sin(np.pi / 180 * angle)
    return dx, dy

def polarise(dx, dy):
    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan(dy / dx)
    return magnitude, angle

def multitext(text, x, y, spacing, font, colour, pos='topleft', antialias=False):
    images = []
    rects = []

    lines = text.split("\n")
    for line in lines:
        image = font.render(line, antialias, colour)
        images.append(image)

        if pos == 'topleft':
            rect = image.get_rect(topleft = (x, y))
        elif pos == 'topright':
            rect = image.get_rect(topright = (x, y))
        elif pos == 'bottomleft':
            rect = image.get_rect(bottomleft = (x, y))
        elif pos == 'bottomright':
            rect = image.get_rect(bottomright = (x, y))
        rects.append(rect)

        y += spacing
    
    return images, rects

def safe_round(x, num_round):
    try:
        return round(x, num_round)
    except:
        return x

def force_round(x, num_round):
    if not isinstance(num_round, int):
        return x
    x = round(x, num_round)
    x_str = str(x)
    try:
        length = len(x_str.split('.')[1])
    except IndexError:
        length = 0
        x_str += '.'
    x_str += '0' * max(0, num_round - length)
    return x_str

def get_dir_path(file):
    return os.path.dirname(os.path.realpath(file))

def shape(val, arr):
    if not isinstance(arr, list):
        return val
    res = []
    for x in arr:
        res.append(shape(val, x))
    return res

def graph(y, xlabel=None, ylabel=None):
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.plot(np.arange(len(y)), y)
    pl.show()
