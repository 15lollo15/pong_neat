from typing import List

import pygame as pg
from pygame.sprite import Group
import settings


class Paddle(pg.sprite.Sprite):
    def __init__(self, pos:tuple, color:tuple = (255,255,255), groups: List[Group] = []) -> None:
        super().__init__(groups)
        self.pos = pos
        self.color = color
        self.height = settings.PADDLE_SIZE
        self.image = self.build_paddle()
        self.rect = self.image.get_rect(center=pos)
        self.is_dead = False
        self.direction = 0
        self.speed = 4

    def build_paddle(self):
        surface = pg.Surface((10, self.height), pg.SRCALPHA)
        pg.draw.rect(surface, self.color, (0, 0, 10, self.height))
        return surface

    def go_up(self):
        self.direction = -1

    def go_down(self):
        self.direction = 1

    def stay(self):
        self.direction = 0

    def update(self, delta_t: float):
        self.rect.center += pg.Vector2(0, self.direction) * self.speed