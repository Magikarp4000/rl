from agents import *
import pygame
from pygame.locals import *
import racetrack_utils
from collections import deque


class Car:
    def __init__(self, x, y):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        self.image.fill((255, 0, 0))
    
    def update(self, x, y):
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))


class Square:
    def __init__(self, x, y, _type):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect(center=(x * SIZE + SIZE / 2, y * SIZE + SIZE / 2))
        if _type == 'track':
            self.image.fill((255, 255, 255))
        elif _type == 'start':
            self.image.fill((0, 0, 255))
        elif _type == 'finish':
            self.image.fill((0, 255, 0))


class Racetrack(MCAgent):
    def __init__(self, track, start, finish, max_spe):
        super().__init__()
        self.config(['actions', 'track', 'start', 'finish', 'max_spe'])
        self.track = track
        self.start = start
        self.finish = finish
        self.max_spe = max_spe
        self.core_init()
    
    def set_state_actions(self):
        self.track_set = set(self.track)
        self.start_set = set(self.start)
        self.finish_set = set(self.finish)
        self.states = [(x, y, xspe, yspe) for x, y in self.track
                       for xspe in range(-self.max_spe, self.max_spe + 1)
                       for yspe in range(-self.max_spe, self.max_spe + 1)]
        self.actions = [[(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
                        for _ in self.states]
        self.remove_infinite_actions()
        self.remove_invalid_actions()
    
    def remove_infinite_actions(self):
        for s, state in enumerate(self.states):
            xspe, yspe = state[:2]
            if xspe == 0 and yspe == 0:
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

    def get_episode(self, fast_terminate=False, no_reset=False):
        seq = []
        start_x, start_y = random.choice(self.start)
        state = (start_x, start_y, 0, 0)
        while state[:2] not in self.finish_set:
            x, y, xspe, yspe = state
            s = self.state_to_index(state)
            a = self.get_action(s)
            dx, dy = self.actions[s][a]
            xspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
            yspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
            oldx, oldy = x, y
            x += xspe
            y += yspe
            if (x, y) not in self.track_set:
                if fast_terminate:
                    seq.append((s, a, -1000))
                    break
                elif no_reset:
                    x, y, xspe, yspe = oldx, oldy, xspe, yspe
                else:
                    x, y, xspe, yspe = self.reset()
            state = (x, y, xspe, yspe)
            reward = -1
            seq.append((s, a, reward))
        return seq
    
    def reset(self):
        x, y = random.choice(self.start)
        return x, y, 0, 0

    def animate(self, fps=None, no_reset=False):
        pygame.init()
        W, H = np.array(self.track).max(axis=0) + 1
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((W * SIZE, H * SIZE))
        state = self.reset()
        car = Car(state[0], state[1])
        squares = []
        for x in range(W):
            for y in range(H):
                _type = ''
                if (x, y) in self.finish_set:
                    _type = 'finish'
                elif (x, y) in self.start_set:
                    _type = 'start'
                elif  (x, y) in self.track_set:
                    _type = 'track'
                squares.append(Square(x, y, _type))
        wins = 0
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
            x, y, xspe, yspe = state
            if (x, y) in self.finish_set:
                wins += 1
                print(f"Wins: {wins}")
                x, y = random.choice(self.start)
                xspe = yspe = 0
                state = (x, y, xspe, yspe)
            else:
                s = self.state_to_index(state)
                dx, dy = self.actions[s][self.pi[s]]
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
            car.update(x, y)
            for square in squares:
                screen.blit(square.image, square.rect)
            screen.blit(car.image, car.rect)
            pygame.display.flip()
            if fps:
                clock.tick(fps)
        pygame.quit()


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

SIZE = 15

track, start, finish = racetrack_utils.get_track()
pro = Racetrack(track, start, finish, 3)
