from utils import *

import numpy as np

import pygame
from pygame.locals import *


WIDTH = 500
HEIGHT = 500
CAR_WIDTH = 20
CAR_HEIGHT = 20

ACCEL = 0.15
ANG_ACCEL = 0.5

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
    def __init__(self, x=None, y=None, spe=0, angle=0, ang_spe=0, max_spe=5, max_ang_spe=5, 
                 friction=0.7, ang_friction=0.2):
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

        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect.x = self.x - CAR_WIDTH / 2
        self.rect.y = self.y - CAR_HEIGHT / 2
        self.image.fill(RED)

    def log(self):
        return f"x {self.x} y {self.y} Speed {self.spe} Angle {self.angle}"

    def update(self, action):
        d_spe, d_angle = action

        self.spe += d_spe - self.friction * ACCEL
        self.spe = np.clip(self.spe, self.min_spe, self.max_spe)
        
        self.ang_spe += d_angle
        raw_ang_spe = abs(self.ang_spe)
        raw_ang_spe = np.clip(raw_ang_spe - self.ang_friction * ANG_ACCEL, 0, None)
        self.ang_spe = raw_ang_spe if self.ang_spe >= 0 else -raw_ang_spe
        self.ang_spe = np.clip(self.ang_spe, self.min_ang_spe, self.max_ang_spe)

        self.angle += self.ang_spe
        self.angle = (self.angle + 360) % 360

        dx, dy = vectorise(self.spe, self.angle)

        self.x += dx
        self.y += dy
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)

        self.rect.x = self.x - CAR_WIDTH / 2
        self.rect.y = self.y - CAR_HEIGHT / 2


def get_user_action():
    pressed = pygame.key.get_pressed()
    d_spe = sum([KEY_MAP['speed'][k] for k in KEY_MAP['speed'] if pressed[k]])
    d_angle = sum([KEY_MAP['angle'][k] for k in KEY_MAP['angle'] if pressed[k]])
    return (d_spe, d_angle)

def main_loop(fps=60, log=False):
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    pro = ProCar()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        
        action = get_user_action()
        if log:
            print(f"{pro.log()} Action {action}")
        
        pro.update(action)
        screen.fill(BLACK)
        screen.blit(pro.image, pro.rect)
        pygame.display.flip()
        clock.tick(fps)
        
    pygame.quit()


main_loop(log=1)
