from random import Random

import pygame as pg
import settings
from ball import Ball
from paddle import Paddle
from population import Population

rng = Random()
population = Population(settings.POPULATION_SIZE, 6, 2)

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode(settings.SCREEN_SIZE)
        self.clock = pg.time.Clock()
        self.is_running = False
        self.balls_g = pg.sprite.Group()
        self.paddles_g = pg.sprite.Group()
        self.balls = []
        self.paddles = []
        self.fill_players()

    def fill_players(self):
        self.balls = []
        self.paddles = []
        for i in range(settings.POPULATION_SIZE):
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            p = Paddle((10, settings.SCREEN_SIZE[1] // 2), color=color, groups=[self.paddles_g])
            b = Ball((settings.SCREEN_SIZE[0] // 2, settings.SCREEN_SIZE[1] // 2), p,
                     pg.Vector2(rng.random(), rng.random()), color=color, groups=[self.balls_g])
            self.balls.append(b)
            self.paddles.append(p)

    def manage_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False
                exit()

    def run(self, pop: Population):
        print(len(pop.species))
        self.is_running = True
        while self.is_running:
            pg.display.set_caption('Generation: ' + str(pop.gen))
            delta_t = self.clock.tick(settings.FRAME_RATE)
            self.screen.fill('black')
            self.manage_events()
            for i, a in enumerate(pop.agents):
                ball = self.balls[i]
                paddle = self.paddles[i]
                x = [ball.rect.centerx / settings.SCREEN_SIZE[0], ball.rect.centery / settings.SCREEN_SIZE[1],
                     ball.direction.x, ball.direction.y,
                     paddle.rect.centerx / settings.SCREEN_SIZE[0], paddle.rect.centery / settings.SCREEN_SIZE[1]]
                y = a.predict(x)
                done_something = False
                if y[0] > 0:
                    paddle.go_down()
                    done_something = True
                if y[1] > 0:
                    paddle.go_up()
                    done_something = True
                if not done_something:
                    paddle.stay()

            self.balls_g.update(delta_t)
            self.balls_g.draw(self.screen)

            self.paddles_g.update(delta_t)
            self.paddles_g.draw(self.screen)

            pg.display.update()
            if len(self.balls_g.sprites()) == 0:
                self.is_running = False

        for i, b in enumerate(self.balls):
            #pop.agents[i].fitness = 2**(b.count_bounce + .00001)
            pop.agents[i].fitness = 2 ** (b.count_bounce + .00001)
            if b.distance < settings.PADDLE_SIZE:
                #pop.agents[i].fitness *= 2
                pop.agents[i].fitness += 1

        #pg.quit()


def fitness(pop:Population):
    game = Game()
    game.run(pop)

if __name__ == '__main__':
    population.evolve(fitness)
