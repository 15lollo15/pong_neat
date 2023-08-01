import math
import random
from typing import List

import pygame as pg
from pygame.sprite import Group

import settings
from paddle import Paddle

rng = random.Random()

class Ball(pg.sprite.Sprite):

    def __init__(self, pos: tuple, paddle: Paddle, direction=pg.Vector2(1, 0), color: tuple = (255, 255, 255), groups: List[Group] = []) -> None:
        super().__init__(groups)
        self.diameter = 10
        self.paddle = paddle
        self.radius = self.diameter // 2
        self.color = color
        self.image = self.create_ball_surface()
        self.rect = self.image.get_rect(center=pos)
        self.direction = direction
        self.speed = 6
        self.is_dead = False
        self.count_bounce = 0
        self.distance = 0

    def create_ball_surface(self):
        surface = pg.Surface((self.diameter, self.diameter), pg.SRCALPHA)
        pg.draw.circle(surface, self.color, (self.radius, self.radius), self.radius)
        return surface

    def update(self, delta_t: float):
        self.rect.center += self.direction.normalize() * self.speed

        if self.rect.right > settings.SCREEN_SIZE[0]:
            self.direction.x = -self.direction.x
            self.direction.y = rng.random() * 2 - 1
            self.rect.right = settings.SCREEN_SIZE[0]

        if self.rect.top < 0:
            self.direction.y = -self.direction.y
            self.rect.top = 0

        if self.rect.bottom > settings.SCREEN_SIZE[1]:
            self.direction.y = -self.direction.y
            self.rect.bottom = settings.SCREEN_SIZE[1]

        if self.rect.colliderect(self.paddle.rect):
            delta = self.rect.centery - self.paddle.rect.top - (settings.PADDLE_SIZE / 2)
            prop = delta / settings.PADDLE_SIZE
            self.direction.x = -self.direction.x
            self.direction.y = -prop
            self.rect.left = self.paddle.rect.right
            self.count_bounce += 1

        if self.rect.right < 0:
            self.is_dead = True
            self.paddle.kill()
            self.distance = math.fabs(self.rect.centery - self.paddle.rect.centerx)
            self.kill()

        if self.direction.magnitude() > 0:
            self.direction = self.direction.normalize()

        if self.direction.y > .9 or self.direction.y < -.9:
            self.direction.y = rng.random() * 2 - 1


