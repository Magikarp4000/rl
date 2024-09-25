import numpy as np
import pygame


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

def safe_round(x, num_round):
    if not isinstance(num_round, int):
        return x
    return round(x, num_round)
