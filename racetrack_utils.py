import agents
import random, os
from collections import deque
import numpy as np
import pygame
from pygame.locals import *


WIDTH = 900
HEIGHT = 550
SIZE = 15
X = WIDTH // SIZE
Y = HEIGHT // SIZE
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
SQUARE_COLOURS = {
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
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class _GUISquare:
    def __init__(self, x, y):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        self.x = x
        self.y = y
        self.state = 'empty'
    
    def _update(self, mouse_state, left_down, right_down):
        if left_down:
            self.state = mouse_state
        elif right_down:
            self.state = 'empty'
    
    def update(self, *args, **kwargs):
        self.image.fill(SQUARE_COLOURS[self.state])
    
    def reset(self):
        self.state = 'empty'


class _CursorSquare():
    def __init__(self):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect()
    
    def update(self, mouse_state, *args, **kwargs):
        x, y = _normalise(pygame.mouse.get_pos())
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        self.image.fill(MOUSE_COLOURS[mouse_state])


class Car:
    def __init__(self, x, y):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect()
        self.rect.center = (x * SIZE + SIZE / 2, y * SIZE + SIZE / 2)
        self.image.fill(RED)
    
    def update(self, x, y):
        self.rect.center = (x * SIZE + SIZE / 2, y * SIZE + SIZE / 2)


class Square:
    def __init__(self, x, y, _type):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        self.image.fill(SQUARE_COLOURS[_type])


class Racetrack(agents.Agent):
    def __init__(self, track, start, finish, max_spe):
        super().__init__()
        self.config(['track', 'start', 'finish', 'max_spe'])
        self.track = track
        self.start = start
        self.finish = finish
        self.max_spe = max_spe
        self.core_init()
    
    def set_state_actions(self):
        self.track_set = set(self.track)
        self.start_set = set(self.start)
        self.finish_set = set(self.finish)
        self.starts = [(x, y, 0, 0) for x, y in self.start]
        self.states = [(x, y, xspe, yspe) for x, y in self.track
                       for xspe in range(-self.max_spe, self.max_spe + 1)
                       for yspe in range(-self.max_spe, self.max_spe + 1)]
        self.actions = [[(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
                        for _ in self.states]
        self.remove_infinite_actions()
        self.remove_invalid_actions()
        self.add_terminal_state()
    
    def remove_infinite_actions(self):
        for s, state in enumerate(self.states):
            xspe, yspe = state[:2]
            if xspe == 0 and yspe == 0 and len(self.actions) > 1:
                self.actions[s].remove((0, 0))

    def remove_invalid_actions(self):
        for s, state in enumerate(self.states):
            x, y, xspe, yspe = state
            to_rem = []
            for action in self.actions[s]:
                nx = x + xspe + action[0]
                ny = y + yspe + action[1]
                if (nx, ny) not in self.track_set:
                    to_rem.append(action)
            for action in to_rem:
                if len(self.actions[s]) > 1:
                    self.actions[s].remove(action)
    
    def add_terminal_state(self):
        self.actions.append([(0, 0)])
    
    def next_state(self, s, a):
        x, y, xspe, yspe = self.states[s]
        dx, dy = self.actions[s][a]
        nxspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
        nyspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
        nx = x + nxspe
        ny = y + nyspe
        reward = -1
        if (nx, ny) in self.finish_set:
            return self.size, reward
        if (nx, ny) not in self.track_set:
            nx, ny, nxspe, nyspe = self.reset()
        new_s = self.state_to_index((nx, ny, nxspe, nyspe))
        return new_s, reward

    def calc_rewards(self):
        delta_x = [0, 0, 1, -1]
        delta_y = [1, -1, 0, 0]
        queue = deque()
        d = {pos: np.inf for pos in self.track}
        for pos in self.finish:
            queue.append(pos)
            d[pos] = 0
        while queue:
            x, y = queue.popleft()
            for dx, dy in zip(delta_x, delta_y):
                nx = x + dx
                ny = y + dy
                if (nx, ny) in self.track_set and d[x, y] + 1 < d[nx, ny]:
                    d[nx, ny] = d[x, y] + 1
                    queue.append((nx, ny))
        return {pos: -d[pos] for pos in d}

    def load_convert(self):
        super().load_convert()
        self.track = [tuple(x) for x in self.track]
        self.start = [tuple(x) for x in self.start]
        self.finish = [tuple(x) for x in self.finish]
    
    def reset(self):
        x, y = random.choice(self.start)
        return x, y, 0, 0

    def animate(self, fps=None, no_reset=False, policy='opt', eps=0.1):
        pygame.init()
        W, H = np.array(self.track).max(axis=0) + 1
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((W * SIZE, H * SIZE))
        state = self.reset()
        car = Car(state[0], state[1])
        squares = []
        for x in range(W):
            for y in range(H):
                _type = 'empty'
                if (x, y) in self.finish_set:
                    _type = 'finish'
                elif (x, y) in self.start_set:
                    _type = 'start'
                elif  (x, y) in self.track_set:
                    _type = 'track'
                squares.append(Square(x, y, _type))
        wins = 0
        num_actions = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    if event.key == K_SPACE:
                        state = self.reset()
                        num_actions = 0
            x, y, xspe, yspe = state
            if (x, y) in self.finish_set:
                wins += 1
                print(f"Wins: {wins}    Num actions: {num_actions}")
                x, y = random.choice(self.start)
                xspe = yspe = 0
                state = (x, y, xspe, yspe)
                num_actions = 0
            else:
                s = self.state_to_index(state)
                dx, dy = self.actions[s][self.pi[s]]
                if policy == 'sub':
                    dx, dy = self.actions[s][self.get_action(s, eps)]
                elif policy =='rand':
                    dx, dy = random.choice(self.actions[s])
                xspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
                yspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
                oldx, oldy = x, y
                x += xspe
                y += yspe
                if (x, y) not in self.track_set:
                    if no_reset:
                        x, y, xspe, yspe = oldx, oldy, xspe, yspe
                    else:
                        x, y, xspe, yspe = self.reset()
                state = (x, y, xspe, yspe)
                num_actions += 1
            car.update(x, y)
            for square in squares:
                screen.blit(square.image, square.rect)
            screen.blit(car.image, car.rect)
            pygame.display.flip()
            if fps:
                clock.tick(fps)
        pygame.quit()


def _normalise(pos):
    return pos[0] // SIZE, pos[1] // SIZE

def _reset(sprites):
    for sprite in sprites:
        if isinstance(sprite, _GUISquare):
            sprite.reset()
    return sprites

def _get_squares(sprites):
    squares = []
    for sprite in sprites.values():
        if isinstance(sprite, _GUISquare):
            squares.append(sprite)
    return squares

def _get_squares_dict(sprites):
    return {(sq.x, sq.y): sq for sq in _get_squares(sprites)}

def _bfs(start, squares, mouse_state):
    delta_x = [0, 0, 1, -1]
    delta_y = [1, -1, 0, 0]
    queue = deque()
    visited = {(x, y): False for x in range(X) for y in range(Y)}
    queue.append(start)
    while queue:
        x, y = queue.popleft()
        if (x, y) not in visited or visited[x, y]:
            continue
        visited[x, y] = True
        for dx, dy in zip(delta_x, delta_y):
            nx = x + dx
            ny = y + dy
            if (nx, ny) in visited and squares[nx, ny].state == squares[start].state:
                queue.append((nx, ny))
    visited_list = []
    for pos in visited:
        if visited[pos]:
            visited_list.append(pos)
    return visited_list

def _gui():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((X * SIZE, Y * SIZE))
    sprites = {(x, y): _GUISquare(x, y) for x in range(X) for y in range(Y)}
    sprites.update({'cursor': _CursorSquare()})
    left_down, right_down = False, False
    mouse_state = 'track'
    mouse_action = 'brush'
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
                if event.key == K_1:
                    mouse_state = 'track'
                elif event.key == K_2:
                    mouse_state = 'start'
                elif event.key == K_3:
                    mouse_state = 'finish'
                if event.key == K_f:
                    mouse_action = 'brush' if mouse_action == 'fill' else 'fill'
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
        # Update state
        x, y = _normalise(pygame.mouse.get_pos())
        to_update = [(x, y)]
        if mouse_action == 'fill':
            to_update = _bfs((x, y), _get_squares_dict(sprites), mouse_state)
        for pos in to_update:
            if pos in sprites:
                sprites[pos]._update(mouse_state, left_down, right_down)
        # Update graphics
        for sprite in sprites.values():
            sprite.update(mouse_state)
        for sprite in sprites.values():
            screen.blit(sprite.image, sprite.rect)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
    return _get_squares(sprites)

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

def create_generic_track(w, h):
    track, start, finish = [], [], []
    for y in range(h):
        track.append((0, y))
        track.append((w - 1, y))
    for x in range(w):
        track.append((x, h - 1))
    for x in range(4, 10):
        for y in range(14):
            track.append((x, y))
        start.append((x, 0))
    for x in range(18, 24):
        for y in range(14):
            track.append((x, y))
        finish.append((x, 0))
    for y in range(7, 14):
        for x in range(10, 18):
            track.append((x, y))
    return track, start, finish
