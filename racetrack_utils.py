import pygame
from pygame.locals import *


WIDTH = 900
HEIGHT = 550
SIZE = 20
X = WIDTH // SIZE
Y = HEIGHT // SIZE
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
STATE_COLOURS = {
    'empty': BLACK,
    'track': WHITE,
    'start': BLUE,
    'finish': GREEN
}
MOUSE_COLOURS = {
    'track': RED,
    'start': BLUE,
    'finish': GREEN
}

class _Square:
    def __init__(self, x, y):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        self.x = x
        self.y = y
        self.state = 'empty'
    
    def update(self, mouse_state, left_down, right_down):
        x, y = _normalise(pygame.mouse.get_pos())
        if (x, y) == (self.x, self.y):
            if left_down:
                self.state = mouse_state
            elif right_down:
                self.state = 'empty'
        self.image.fill(STATE_COLOURS[self.state])
    
    def reset(self):
        self.state = 'empty'


class _CursorSquare():
    def __init__(self):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect()
    
    def update(self, mouse_state, *args):
        x, y = _normalise(pygame.mouse.get_pos())
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        self.image.fill(MOUSE_COLOURS[mouse_state])


def _normalise(pos):
    return pos[0] // SIZE, pos[1] // SIZE

def _reset(sprites):
    for sprite in sprites:
        if isinstance(sprite, _Square):
            sprite.reset()
    return sprites

def _gui():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((X * SIZE, Y * SIZE))
    sprites = [_Square(x, y) for x in range(X) for y in range(Y)]
    sprites.append(_CursorSquare())
    left_down, right_down = False, False
    mouse_state = 'track'
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_RETURN:
                    running = False
                if event.key == K_SPACE:
                    sprites = _reset(sprites)
            if event.type == KEYDOWN:
                if event.key == K_1:
                    mouse_state = 'track'
                elif event.key == K_2:
                    mouse_state = 'start'
                elif event.key == K_3:
                    mouse_state = 'finish'
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    left_down = True
                elif event.button == 3:
                    right_down = True
            if event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    left_down = False
                elif event.button == 3:
                    right_down = False
        for sprite in sprites:
            sprite.update(mouse_state, left_down, right_down)
        for sprite in sprites:
            screen.blit(sprite.image, sprite.rect)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
    squares = []
    for sprite in sprites:
        if isinstance(sprite, _Square):
            squares.append(sprite)
    return squares

def get_track():
    squares = _gui()
    track, start, finish = [], [], []
    for square in squares:
        pos = (square.x, square.y)
        if square.state == 'track':
            track.append(pos)
        elif square.state == 'start':
            start.append(pos)
            track.append(pos)
        elif square.state == 'finish':
            finish.append(pos)
            track.append(pos)
    track = list(dict.fromkeys(track))
    return track, start, finish
