import numpy as np
import pygame


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class TileCoding:
    def __init__(self, n: int, dim: int, bounds: list, num_per_dim: list, offsets: list):
        assert(len(bounds) == dim)
        assert(len(num_per_dim) == dim)
        assert(len(offsets) == dim)

        for cur_bounds in bounds:
            if isinstance(cur_bounds, int):
                cur_bounds = [cur_bounds]
            assert(len(cur_bounds) == 2)
        
        self.n = n
        self.dim = dim
        self.bounds = bounds
        self.num_per_dim = num_per_dim
        self.offsets = offsets
        self.total_per_layer = np.prod(np.array(self.num_per_dim) + 1)

    def encode(self, state: list):
        assert(len(state) == self.dim)

        res = []
        for i in range(self.n):
            encoding = i * self.total_per_layer
            multiplier = 1

            for j in range(self.dim):
                cur_bounds = self.bounds[j]

                length = cur_bounds[1] - cur_bounds[0]
                size = length / self.num_per_dim[j]

                offset = i * size * self.offsets[j] / self.n
                cur_raw = (state[j] + offset - cur_bounds[0]) / size
                cur = int(np.floor(cur_raw))

                encoding += cur * multiplier
                multiplier *= self.num_per_dim[j] + 1
            res.append(encoding)
        return res

    def decode(self, encoding: list):
        decoding = []
        for code in encoding:
            mod_code = code % self.total_per_layer
            cur_decoding = []
            for num in self.num_per_dim:
                mod_code, value = divmod(mod_code, num)
                cur_decoding.append(value)
            decoding.append(cur_decoding)
        return decoding


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

def multitext(text, x, y, spacing, font, colour, antialias=False):
    images = []
    rects = []

    lines = text.split("\n")
    for line in lines:
        image = font.render(line, antialias, colour)
        images.append(image)

        rect = image.get_rect(topleft = (x, y))
        rects.append(rect)

        y += spacing
    
    return images, rects

tile = TileCoding(4, 2, [(0, 8), (-1, 1)], [4, 4], [1, 1])
code = tile.encode([2.5, -0.7])
print(tile.decode(code))