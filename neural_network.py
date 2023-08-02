import math
from typing import List, Dict

import pygame as pg
from neat_py.genome import Genome


class NeuralNetwork(pg.sprite.Sprite):
    def __init__(self, neural_network: Genome, pos: tuple, size: tuple, groups: List[pg.sprite.Group] = []) -> None:
        super().__init__(groups)
        self.neural_network: Genome = neural_network
        self.size = size
        self.image = self.draw_neural_network()
        self.rect = self.image.get_rect(topleft=pos)

    def draw_neural_network(self) -> pg.Surface:
        surface = pg.Surface(self.size, pg.SRCALPHA)
        nodes_per_layer: Dict[int, int] = self.neural_network.compute_nodes_per_layer()
        counter_per_layer: Dict[int, int] = {}


        padding_x: int = self.size[0] // (self.neural_network.layers + 1)
        paddings_y: Dict[int, int] = {}

        for layer in nodes_per_layer.keys():
            paddings_y[layer] = self.size[1] // (nodes_per_layer[layer] + 1)
            counter_per_layer[layer] = 0

        pos_dict: Dict[int, tuple] = {}
        for node in self.neural_network.nodes:
            x = padding_x * (node.layer + 1)
            y = paddings_y[node.layer] * (counter_per_layer[node.layer] + 1)
            counter_per_layer[node.layer] = counter_per_layer[node.layer] + 1
            pos_dict[node.number] = (x, y)

        max_weight = 0
        for connection in self.neural_network.connections:
            if connection.enabled and math.fabs(connection.weight) > max_weight:
                max_weight = math.fabs(connection.weight)

        for connection in self.neural_network.connections:
            from_node = connection.from_node
            to_node = connection.to_node
            start_point = pos_dict[from_node.number]
            end_point = pos_dict[to_node.number]
            if connection.weight >= 0:
                g = int(255 * (connection.weight / max_weight))
                g = max(0, g)
                g = min(255, g)
                color = (0, g, 0)
                pg.draw.line(surface, color, start_point, end_point)
            else:
                b = int(255 * (math.fabs(connection.weight) / max_weight))
                b = max(0, b)
                b = min(255, b)
                color = (0, 0, b)
                pg.draw.line(surface, color, start_point, end_point)

        for node in self.neural_network.nodes:
            pos = pos_dict[node.number]
            pg.draw.circle(surface, 'red', pos, 10)

        return surface
