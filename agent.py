from typing import List

from genome import Genome
from neat_settings import EXCEED_PENALTY, DISJOINT_PENALTY, WEIGHT_DIFFERENCE_PENALTY


class Agent:
    def __init__(self, agent_id: int, num_input: int, num_output: int, off_spring: bool = False) -> None:
        self.agent_id: int = agent_id
        self.num_input: int = num_input
        self.num_output: int = num_output

        self.fitness: float = .0
        self.brain: Genome = Genome(agent_id, num_input, num_output, off_spring=off_spring)

    def predict(self, x: List[float]) -> List[float]:
        return self.brain.predict(x)

    def mutate(self) -> None:
        self.brain.mutate()

    def crossover(self, agent: 'Agent', child_id: int = 0) -> 'Agent':
        child: Agent = Agent(child_id, self.num_input, self.num_output, off_spring=True)
        if agent.fitness < self.fitness:
            child.brain = self.brain.crossover(agent.brain)
        else:
            child.brain = agent.brain.crossover(self.brain)
        return child

    def similarity(self, agent: 'Agent', n: int):
        agent_brain: Genome = agent.brain
        return self.brain.similarity(agent_brain, n, EXCEED_PENALTY, DISJOINT_PENALTY, WEIGHT_DIFFERENCE_PENALTY)

    def clone(self) -> 'Agent':
        clone: 'Agent' = Agent(self.agent_id, self.num_input, self.num_output, off_spring=True)
        clone.fitness = self.fitness
        clone.brain = self.brain.clone()
        return clone
