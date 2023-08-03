from typing import List, Tuple

import pygame as pg
from pygame.sprite import Group
import settings


class Paddle(pg.sprite.Sprite):
    def __init__(self, pos: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255),
                 groups: List[Group] = []) -> None:
        super().__init__(groups)
        self.pos: Tuple[int, int] = pos
        self.color: Tuple[int, int, int] = color
        self.height: int = settings.PADDLE_SIZE
        self.image: pg.Surface = self.build_paddle()
        self.rect: pg.Rect = self.image.get_rect(center=pos)
        self.is_dead: bool = False
        self.direction: int = 0
        self.speed: int = 4

    def build_paddle(self) -> pg.Surface:
        surface = pg.Surface((10, self.height), pg.SRCALPHA)
        pg.draw.rect(surface, self.color, (0, 0, 10, self.height))
        return surface

    def go_up(self) -> None:
        self.direction = -1

    def go_down(self) -> None:
        self.direction = 1

    def stay(self) -> None:
        self.direction = 0

    def update(self, delta_t: float) -> None:
        self.rect.center += pg.Vector2(0, self.direction) * self.speed