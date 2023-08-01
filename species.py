from typing import List

from agent import Agent
from neat_settings import rng


class Species:
    def __init__(self, rep: Agent) -> None:
        super().__init__()
        self.rep: Agent = rep
        self.agents: List[Agent] = []
        self.avg_fitness: float = 0
        self.generation_without_improvement: int = 0
        self.champion: Agent = rep
        self.old_fitness: float = 0
        self.fitness = 0

    def cut(self) -> None:
        if len(self.agents) >= 2:
            self.agents = self.agents[:len(self.agents) // 2]

    def pull_child(self) -> Agent:
        weights: List[float] = [a.fitness for a in self.agents]
        parent_1: Agent = rng.choices(self.agents, weights=weights)[0]
        parent_2: Agent = rng.choices(self.agents, weights=weights)[0]
        child = parent_1.crossover(parent_2)
        child.mutate()
        return child

    def is_empty(self) -> bool:
        return len(self.agents) == 0

    def update_fitness(self) -> float:
        self.sort()
        self.champion = self.agents[0]
        self.fitness = self.champion.fitness
        return self.fitness

    def update_old_fitness(self):
        now_fitness = self.champion.fitness
        delta = now_fitness - self.old_fitness
        if delta < 0:
            self.generation_without_improvement += 1
        else:
            self.generation_without_improvement = 0
        self.old_fitness = now_fitness

    def reset(self) -> None:
        self.agents = []

    def sort(self) -> None:
        self.agents.sort(key=(lambda a: a.fitness), reverse=True)

    def update_avg(self) -> float:
        fitness_sum: float = 0
        for agent in self.agents:
            fitness_sum += agent.fitness
        self.avg_fitness = fitness_sum / len(self.agents)
        return self.avg_fitness

    def apply_fitness_sharing(self) -> None:
        for agent in self.agents:
            agent.fitness /= len(self.agents)

