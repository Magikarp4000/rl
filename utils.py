import os
import numpy as np
import matplotlib.pyplot as pl


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class StepData:
    def __init__(self, s, a, r, step_num, ep_num):
        self.s = s
        self.a = a
        self.r = r
        self.step_num = step_num
        self.ep_num = ep_num
        self.sar = (self.s, self.a, self.r)


class Buffer:
    def __init__(self, size, default=None):
        self.size = size
        self.default = default
        self._buffer = [default for _ in range(size)]
        self.idx = 0
    
    def __call__(self):
        return self._buffer
    
    def get(self, idx):
        return self._buffer[idx % self.size]
    
    def set(self, idx, val):
        self._buffer[idx % self.size] = val
    
    def reset(self):
        self._buffer = [self.default for _ in range(self.size)]
    
    def reset_idx(self, idx=0):
        self.idx = idx

    def update(self, val):
        self.set(self.idx, val)
        self.idx = (self.idx + 1) % self.size


class VariableBuffer(Buffer):
    def __init__(self, size, default=None):
        super().__init__(size, default)
        self._buffer = []
    
    def set(self, idx, val):
        try:
            self._buffer[idx % self.size] = val
        except IndexError:
            self._buffer.append(val)
    
    def cur_size(self):
        return len(self._buffer)


class SingleBuffer(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep = 0
    
    def set_ep(self, ep):
        self.ep = ep


class ReplayBuffer:
    def __init__(self):
        self._buf1 = SingleBuffer()
        self._buf2 = SingleBuffer()
        self._write = self._buf1
        self._read = self._buf1
        self._read_idx = 0
    
    def read(self):
        if self._read_idx < len(self._read):
            return self._get_data()
        self._reset_read()
        if self._read:
            return self._get_data()
        return None
    
    def next(self):
        if self._read_idx < len(self._read):
            self._read_idx += 1
        else:
            self._reset_read()

    def write(self, val):
        self._write.append(val)
    
    def clear(self):
        self._write.clear()
    
    def create_new_ep(self, ep):
        if self._write is self._read and len(self._write) > 0:
            self._swap_write()
        self._write.clear()
        self._write.set_ep(ep)
    
    def _get_data(self):
        return StepData(*self._read[self._read_idx], self._read_idx, self._read.ep)

    def _reset_read(self):
        self._read = self._write
        self._read_idx = 0

    def _swap_write(self):
        if self._write == self._buf1:
            self._write = self._buf2
        else:
            self._write = self._buf1


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

def get_dir(file=__file__):
    return os.path.dirname(os.path.realpath(file))

def flatten(arr):
    if not isinstance(arr, list):
        return [arr]
    res = []
    for x in arr:
        res.extend(flatten(x))
    return res

def fit_shape(val, arr):
    if not isinstance(arr, list):
        return val
    res = []
    for x in arr:
        res.append(fit_shape(val, x))
    return res

def graph(y, xlabel=None, ylabel=None):
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.plot(np.arange(len(y)), y)
    pl.show()

def name(obj):
    return type(obj).__name__.lower()
