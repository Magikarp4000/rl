from agents import *
import pygame
from pygame.locals import *


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
        self.config(['track', 'start', 'finish', 'max_spe', 'n'])
        self.max_spe = max_spe
        self.track = track
        self.start = start
        self.finish = finish
        self.track_set = set(self.track)
        self.start_set = set(self.start)
        self.finish_set = set(self.finish)
        self.W, self.H = np.array(self.track).max(axis=0) + 1
        self.core_init()
    
    def set_state_actions(self):
        self.states = [(x, y, xspe, yspe) for x, y in self.track
                       for xspe in range(-self.max_spe, self.max_spe + 1)
                       for yspe in range(-self.max_spe, self.max_spe + 1)]
        self.actions = [[(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
                        for _ in self.states]
        for s, state in enumerate(self.states):
            xspe, yspe = state[2:]
            if xspe == 0 and yspe == 0:
                self.actions[s].remove((0, 0))
    
    def get_episode(self):
        seq = []
        start_x, start_y = random.choice(self.start)
        state = (start_x, start_y, 0, 0)
        while state[:2] not in self.finish_set:
            x, y, xspe, yspe = state
            s = self.state_to_index(state)
            a = self.get_action(s)
            # print(self.actions[s][a])
            dx, dy = self.actions[s][a]
            xspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
            yspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
            oldx, oldy = x, y
            x = max(0, min(self.W - 1, x + xspe))
            y = max(0, min(self.H - 1, y + yspe))
            if (x, y) not in self.track_set:
                state = self.reset()
            else:
                state = (x, y, xspe, yspe)
            reward = -1
            seq.append((s, a, reward))
            # print(state)
        return seq
    
    def reset(self):
        x, y = random.choice(self.start)
        return x, y, 0, 0

    def animate(self, fps=None):
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.W * SIZE, self.H * SIZE))
        state = self.reset()
        car = Car(state[0], state[1])
        squares = []
        for y in range(self.H):
            for x in range(self.W):
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
                # dx, dy = 0, 0
                xspe = max(-self.max_spe, min(self.max_spe, xspe + dx))
                yspe = max(-self.max_spe, min(self.max_spe, yspe + dy))
                oldx, oldy = x, y
                x = max(0, min(self.W - 1, x + xspe))
                y = max(0, min(self.H - 1, y + yspe))
                if (x, y) not in self.track_set:
                    state = self.reset()
                else:
                    state = (x, y, xspe, yspe)
            car.update(x, y)
            for square in squares:
                screen.blit(square.image, square.rect)
            screen.blit(car.image, car.rect)
            pygame.display.flip()
            if fps:
                clock.tick(fps)
        pygame.quit()


def create_track(w, h):
    track, start, finish = [], [], []
    for y in range(h):
        track.append((0, y))
        track.append((w - 1, y))
    for x in range(w):
        track.append((x, h - 1))
    for x in range(4, 8):
        for y in range(10):
            track.append((x, y))
        start.append((x, 0))
    for x in range(12, 16):
        for y in range(10):
            track.append((x, y))
        finish.append((x, 0))
    for y in range(6, 10):
        for x in range(8, 12):
            track.append((x, y))
        # finish.append((w - 1, y))
    return track, start, finish

SIZE = 20
W = 20
H = 20
track, start, finish = create_track(W, H)
pro = Racetrack(track, start, finish, 3)
init_pi = [5 for _ in pro.states]
pro.train('onpolicy', 1000, eps=0.5, pi=init_pi, batch_size=100)
# pro.train('offpolicy', 50000, batch_size=1000)
pro.animate(4)
# print([pro.actions[s][pro.pi[s]] for s in range(pro.size)])
# print(pro.q)