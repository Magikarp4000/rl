from agents import *
from utils import *

import numpy as np

import pygame
from pygame.locals import *


WIDTH = 500
HEIGHT = 300
CAR_WIDTH = 20
CAR_HEIGHT = 20


class CarDisplay():
    def __init__(self, x):
        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect.x = x * WIDTH - CAR_WIDTH / 2
        self.rect.y = HEIGHT / 2 - CAR_HEIGHT / 2
        self.image.fill(RED)
    
    def update(self, x):
        if x != TERMINAL:
            self.rect.x = x * WIDTH - CAR_WIDTH / 2
            y = normalise_bounds(np.sin(np.pi * x), (0, 1))
            self.rect.y = y * HEIGHT - CAR_HEIGHT / 2


class Car(Approximator):
    def __init__(self, base_actions=[-1, 0, 1], bounds=[(-1, 1), (-1, 1)], start_bounds=[(-1, 1), (0, 0)]):
        super().__init__(base_actions, bounds, start_bounds, dim=2)
        # User play
        self.move_map = {
            K_LEFT: -1,
            K_RIGHT: 1
        }
    
    def next_state(self, s, a=None, action=None):
        if s == TERMINAL:
            return TERMINAL, TERMINAL_REWARD
        if a is not None:
            action = self.base_actions[a]
        pos, vel = s
        new_vel = np.clip(vel + 0.001 * action - 0.0025 * np.cos(3 * pos), self.bounds[1][0], self.bounds[1][1])
        new_pos = np.clip(pos + new_vel, self.bounds[0][0], self.bounds[0][1])
        if new_pos == self.bounds[0][0]:
            new_vel = 0
        new_s = [new_pos, new_vel]
        if new_pos == self.bounds[0][1]:
            new_s = TERMINAL
        reward = -1
        return new_s, reward

    def set_state_actions(self):
        return

    def normalise(self, s):
        if s == TERMINAL:
            return TERMINAL
        return normalise_bounds(s[0], self.bounds[0])

    def animate(self, fps=60, mode='AI', log=False):
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        
        s = self.init_state()
        init_s = s.copy()
        disp = CarDisplay(self.normalise(s))

        running = True
        steps = 0
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    if event.key == K_SPACE:
                        s = init_s
                        steps = 0
            
            if s == TERMINAL:
                running = False
            
            if mode == 'AI':
                a = self.get_action(s, eps=0)
                if log:
                    print(s, self.base_actions[a])
                s, _ = self.next_state(s, a)
            elif mode == 'user':
                action = self.get_user_action()
                if log:
                    print(s, action)
                s, _ = self.next_state(s, action=action)
            steps += 1

            screen.fill(BLACK)
            disp.update(self.normalise(s))
            screen.blit(disp.image, disp.rect)
            pygame.display.flip()
            clock.tick(fps)
        
        pygame.quit()
        print(f"Steps taken: {steps}")

    def get_user_action(self):
        pressed = pygame.key.get_pressed()
        move = sum([self.move_map[k] for k in self.move_map if pressed[k]])
        return move
    
    def simulate(self, fps=60, log=False):
        self.animate(fps, 'AI', log)
    
    def play(self, fps=60, log=False):
        self.animate(fps, 'user', log)


car = Car(bounds=[(-1.2, 0.5), (-0.07, 0.07)], start_bounds=[(-0.6, -0.4), (0, 0)])
car.load('mountain/v1.0')
car.train('sarsa', num_ep=100, gamma=1.0, alpha=0.4, eps=0.1, lamd=0.85, num_layers=8, num_per_dim=[8, 8], 
          offsets=[1, 3], batch_size=10)
# car.save('mountain/v2.1')
print(car.test(100))
car.simulate()
# car.play()
# car.train('sarsa', 0, num_tiles=8, tile_frac=8)
# print(car.get_tile_coding([-0.9879, -0.0525]))
