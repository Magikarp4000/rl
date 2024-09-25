from utils import *
from agents import *

import numpy as np
import itertools

import pygame
from pygame.locals import *


pygame.init()
info = pygame.display.Info()

WIDTH = 500
HEIGHT = 500
PADDING = 20

pygame.font.init()
FONT_SIZE = 15
FONT = pygame.font.SysFont('Arial', FONT_SIZE)

BG_COLOUR = WHITE

CAR_WIDTH = 30
CAR_HEIGHT = 30

ACCEL = 0.15
ANG_ACCEL = 0.4

FRICTION = 0.7
ANG_FRICTION = 0.7

KEY_MAP = {
    'angle': {
        K_LEFT: ANG_ACCEL,
        K_RIGHT: -ANG_ACCEL
    },
    'speed': {
        K_UP: ACCEL
    }
}


class ProCar():
    def __init__(self, x=None, y=None, spe=0, angle=0, ang_spe=0, max_spe=7.5, max_ang_spe=5, 
                 friction=0.7, ang_friction=0.7):
        self.x = WIDTH / 2 if x is None else x
        self.y = HEIGHT / 2 if y is None else y
        self.spe = spe
        self.angle = angle
        self.ang_spe = ang_spe
        self.min_spe = 0
        self.max_spe = max_spe
        self.min_ang_spe = -max_ang_spe
        self.max_ang_spe = max_ang_spe
        self.friction = friction
        self.ang_friction = ang_friction

        self.image = pygame.image.load("racetrack/assets/car.png").convert_alpha()
        self.image = pygame.transform.flip(self.image, False, True)
        self.image = pygame.transform.scale(self.image, (CAR_WIDTH, CAR_HEIGHT))
        self.clean_image = self.image.copy()
        
        self.rect = self.image.get_rect()
        self.rect.x = self.x - CAR_WIDTH / 2
        self.rect.y = self.y - CAR_HEIGHT / 2
    
    def get_data(self, num_round=None):
        data = {
            'x': safe_round(self.x, num_round),
            'y': safe_round(self.y, num_round),
            'Speed': safe_round(self.spe, num_round),
            'Angle': safe_round(self.angle, num_round),
            'Angular velocity': safe_round(self.ang_spe, num_round)
        }
        return data
    
    def log(self, num_round=None):
        data = self.get_data(num_round)
        res = ""
        for key in data:
            res += f"{key} {data[key]} "
        return res.strip()

    def update(self, action):
        accel, ang_accel = action

        self.spe += accel - self.friction * ACCEL
        self.spe = np.clip(self.spe, self.min_spe, self.max_spe)
        
        self.ang_spe += ang_accel
        raw_ang_spe = abs(self.ang_spe)
        raw_ang_spe = np.clip(raw_ang_spe - self.ang_friction * ANG_ACCEL, 0, None)
        self.ang_spe = raw_ang_spe if self.ang_spe >= 0 else -raw_ang_spe
        self.ang_spe = np.clip(self.ang_spe, self.min_ang_spe, self.max_ang_spe)

        self.angle += self.ang_spe
        self.angle = (self.angle + 360) % 360

        dx, dy = vectorise(self.spe, 90 - self.angle)

        self.x += dx
        self.y += dy
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)

        self.image = pygame.transform.rotate(self.clean_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))


class Text:
    def __init__(self):
        self.images = []
        self.rects = []
    
    def update(self, data):
        text = ""
        for key in data:
            text += f"{key}: {data[key]}\n"
        self.images, self.rects = multitext(text, PADDING, PADDING, PADDING, FONT, BLACK)


class Game:
    def __init__(self, width, height, bg_colour):
        self.width = width
        self.height = height
        self.bg_colour = bg_colour

    def get_user_action(self):
        pressed = pygame.key.get_pressed()
        accel = sum([KEY_MAP['speed'][k] for k in KEY_MAP['speed'] if pressed[k]])
        ang_accel = sum([KEY_MAP['angle'][k] for k in KEY_MAP['angle'] if pressed[k]])
        return (accel, ang_accel)

    def get_info(self, sprite, clock, num_round=None):
        info = {
            'FPS': safe_round(clock.get_fps(), num_round)
        }
        info.update(sprite.get_data(num_round))
        return info

    def main(self, fps=60, log=False):
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()

        pro = ProCar()
        pro_text = Text()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    if event.key == K_f:
                        log = not log
            
            action = self.get_user_action()

            if log:
                print(f"{pro.log(2)} Action {action} FPS {clock.get_fps()}")

            pro.update(action)
            pro_text.update(self.get_info(pro, clock, 2))

            screen.fill(self.bg_colour)
            
            screen.blit(pro.image, pro.rect)
            for image, rect in zip(pro_text.images, pro_text.rects):
                screen.blit(image, rect)
            pygame.display.flip()
            
            clock.tick(fps)
            
        pygame.quit()


class ProtrackModel(Approximator):
    def __init__(self, accel, ang_accel, friction, ang_friction, bounds, start_bounds):
        base_actions = list(itertools.product([accel, 0], [-ang_accel, 0, ang_accel]))
        super().__init__(base_actions, bounds, start_bounds, 5)

        self.config(['accel', 'ang_accel', 'friction', 'ang_friction'])
        self.accel = accel
        self.ang_accel = ang_accel
        self.friction = friction
        self.ang_friction = ang_friction
    
    def set_state_actions(self):
        return

    def next_state(self, state, a=None, action=None):
        if state == TERMINAL:
            return TERMINAL, TERMINAL_REWARD
        if a is not None:
            action = self.base_actions[a]
        
        x, y, spe, angle, ang_spe = state
        accel, ang_accel = action

        spe += accel - self.friction * ACCEL
        spe = np.clip(spe, self.bounds[2][0], self.bounds[2][1])
        
        ang_spe += ang_accel
        raw_ang_spe = abs(ang_spe)
        raw_ang_spe = np.clip(raw_ang_spe - self.ang_friction * ANG_ACCEL, 0, None)
        ang_spe = raw_ang_spe if ang_spe >= 0 else -raw_ang_spe
        ang_spe = np.clip(ang_spe, self.bounds[4][0], self.bounds[4][1])

        angle += ang_spe
        angle = (angle + 360) % 360

        dx, dy = vectorise(spe, 90 - angle)

        x += dx
        y += dy
        x = np.clip(x, self.bounds[0][0], self.bounds[0][1])
        y = np.clip(y, self.bounds[1][0], self.bounds[1][1])

        new_state = (x, y, spe, angle, ang_spe)
        reward = -1
        if x >= self.bounds[0][1]:
            new_state = TERMINAL
        
        return new_state, reward



# game = Game(WIDTH, HEIGHT, BG_COLOUR)
# game.main()

model = ProtrackModel(ACCEL, ANG_ACCEL, FRICTION, ANG_FRICTION,
                      [[0, WIDTH], [0, HEIGHT], [0, 7.5], [0, 360], [-5, 5]],
                      [[0, 0], [0, HEIGHT], [0, 0], [0, 0], [0, 0]],
                      )
model.train('sarsa', 500, num_per_dim=[8, 8, 8, 8, 8], offsets=[1, 3, 5, 7, 9])
