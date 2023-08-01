import math
from typing import List, Dict, Any

from connection import Connection
from node import Node
from neat_settings import rng, AF_BIAS_PARTNER_INHERIT_PROBABILITY, \
    CONNECTIONS_PARTNER_INHERIT_PROBABILITY, WEIGHT_MUTATION_PROBABILITY, BIAS_MUTATION_PROBABILITY, \
    ACTIVATION_MUTATION_PROBABILITY, NEW_CONNECTION_PROBABILITY, NEW_NODE_PROBABILITY


class Genome:

    def __init__(self, genome_id: int, inputs: int, outputs: int, off_spring: bool = False) -> None:
        self.genome_id: int = genome_id
        self.inputs: int = inputs
        self.outputs: int = outputs

        self.layers: int = 2
        self.next_node: int = 0

        self.nodes: List[Node] = []
        self.connections: List[Connection] = []

        if not off_spring:
            self.build()

    def build(self) -> None:
        for _ in range(self.inputs):
            self.nodes.append(Node(self.next_node, 0))
            self.next_node += 1

        for _ in range(self.outputs):
            self.nodes.append(Node(self.next_node, 1, is_output=True))
            self.next_node += 1

        for i in range(self.inputs):
            for j in range(self.inputs, self.inputs + self.outputs):
                weight: float = rng.random() * self.inputs * math.sqrt(2 / self.inputs)
                self.connections.append(Connection(self.nodes[i], self.nodes[j], weight))

    def sort_by_layer(self) -> None:
        self.nodes.sort(key=lambda n: n.layer)

    def generate_network(self) -> None:
        for node in self.nodes:
            node.output_connections = []

        for connection in self.connections:
            connection.from_node.output_connections.append(connection)

        self.sort_by_layer()

    def predict(self, input_values: List[float]) -> List[float]:
        self.generate_network()

        for node in self.nodes:
            node.input_sum = 0

        for i in range(self.inputs):
            self.nodes[i].output_value = input_values[i]

        result: List[float] = []
        for node in self.nodes:
            node.predict()
            if node.is_output:
                result.append(node.output_value)

        return result

    def get_node(self, node_number: int) -> int:
        for i in range(len(self.nodes)):
            if self.nodes[i].number == node_number:
                return i

        return -1

    @staticmethod
    def common_connection(inn_n: int, connections: List[Connection]) -> int:
        for i in range(len(connections)):
            if inn_n == connections[i].get_innovation_number():
                return i
        return -1

    def init_off_spring(self, partner: 'Genome') -> 'Genome':
        off_spring: Genome = Genome(0, self.inputs, self.outputs, off_spring=True)
        off_spring.next_node = self.next_node

        for i in range(len(self.nodes)):
            node: Node = self.nodes[i].clone()
            if node.is_output:
                partner_node: Node = partner.nodes[partner.get_node(node.number)]
                if rng.random() > AF_BIAS_PARTNER_INHERIT_PROBABILITY:
                    node.activation = partner_node.activation
                    node.bias = partner_node.bias
            off_spring.nodes.append(node)
        return off_spring

    def crossover(self, partner: 'Genome') -> 'Genome':
        off_spring: Genome = self.init_off_spring(partner)

        for i in range(len(self.connections)):
            index: int = self.common_connection(self.connections[i].get_innovation_number(), partner.connections)

            if index != -1:
                conn = self.connections[i].clone() if rng.random() > CONNECTIONS_PARTNER_INHERIT_PROBABILITY else \
                    partner.connections[index].clone()
            else:
                conn: Connection = self.connections[i].clone()

            from_node: Node = off_spring.nodes[off_spring.get_node(conn.from_node.number)]
            to_node: Node = off_spring.nodes[off_spring.get_node(conn.to_node.number)]
            conn.from_node = from_node
            conn.to_node = to_node

            if from_node is not None and to_node is not None:
                off_spring.connections.append(conn)

        off_spring.layers = self.layers

        return off_spring

    def add_node(self) -> None:
        picked_connection: Connection = rng.choice(self.connections)
        picked_connection.enabled = False
        self.connections.remove(picked_connection)

        new_node: Node = Node(self.next_node, picked_connection.from_node.layer + 1)
        for node in self.nodes:
            if node.layer > picked_connection.from_node.layer:
                node.layer += 1

        new_connection1: Connection = Connection(picked_connection.from_node, new_node, 1)
        new_connection2: Connection = Connection(new_node, picked_connection.to_node, picked_connection.weight)

        self.layers += 1
        self.connections.append(new_connection1)
        self.connections.append(new_connection2)
        self.nodes.append(new_node)
        self.next_node += 1

        if self.layers == 2 and len(self.nodes) > 3:
            raise Exception('aj capit')

    def nodes_connected(self, node1: Node, node2: Node) -> bool:
        for i in range(len(self.connections)):
            conn: Connection = self.connections[i]
            if (conn.from_node == node1 and conn.to_node == node2) or (
                    conn.from_node == node2 and conn.to_node == node1):
                return True

        return False

    def fully_connected(self) -> bool:
        max_connections: int = 0
        nodes_per_layer: Dict[int, int] = {}

        for node in self.nodes:
            if nodes_per_layer.get(node.layer) is not None:
                nodes_per_layer[node.layer] += 1
            else:
                nodes_per_layer[node.layer] = 1

        #for i in range(self.layers - 1):
        for i in range(self.layers):
            for j in range(i + 1, self.layers):
                max_connections += nodes_per_layer[i] * nodes_per_layer[j]

        return max_connections == len(self.connections)

    def add_connection(self) -> None:
        if self.fully_connected():
            return
        node1: int = rng.randint(0, len(self.nodes) - 1)
        node2: int = rng.randint(0, len(self.nodes) - 1)

        while self.nodes[node1].layer == self.nodes[node2].layer or self.nodes_connected(self.nodes[node1], self.nodes[node2]):
            node1 = rng.randint(0, len(self.nodes) - 1)
            node2 = rng.randint(0, len(self.nodes) - 1)

        if self.nodes[node1].layer > self.nodes[node2].layer:
            temp = node1
            node1 = node2
            node2 = temp

        new_connection = Connection(self.nodes[node1], self.nodes[node2],
                                    rng.random() * self.inputs * math.sqrt(2 / self.inputs))
        self.connections.append(new_connection)

    def mutate(self) -> None:
        if rng.random() < WEIGHT_MUTATION_PROBABILITY:
            for conn in self.connections:
                conn.mutate_weight()

        if rng.random() < BIAS_MUTATION_PROBABILITY:
            for node in self.nodes:
                node.mutate_bias()

        if rng.random() < ACTIVATION_MUTATION_PROBABILITY:
            i = rng.randint(0, len(self.nodes) - 1)
            self.nodes[i].mutate_activation()

        if rng.random() < NEW_CONNECTION_PROBABILITY:
            self.add_connection()

        if rng.random() < NEW_NODE_PROBABILITY:
            self.add_node()

    def clone(self) -> 'Genome':
        clone: Genome = Genome(self.genome_id, self.inputs, self.outputs)
        clone.nodes = self.nodes.copy()
        clone.connections = self.connections.copy()
        clone.layers = self.layers
        clone.next_node = self.next_node

        return clone

    def calculate_weight(self) -> int:
        return len(self.connections) + len(self.nodes)

    @staticmethod
    def get_last_elem(array: []) -> Any:
        return array[len(array) - 1]

    def similarity(self, other: 'Genome', n: int, exceed_penalty: float, disjoint_penalty: float, w_diff_penalty: float) -> float:
        this_connections = self.connections.copy()
        other_connections = other.connections.copy()
        this_connections.sort(key=(lambda c: c.get_innovation_number()))
        other_connections.sort(key=(lambda c: c.get_innovation_number()))

        this_max_innovation = self.get_last_elem(this_connections).get_innovation_number()
        other_max_innovation = self.get_last_elem(other_connections).get_innovation_number()
        max_innovations = max(this_max_innovation, other_max_innovation)

        this_index = 0
        other_index = 0

        exceed = 0
        disjoint = 0
        w_diff = 0
        for now_innovation in range(max_innovations + 1):
            while this_index < len(this_connections) and \
                    this_connections[this_index].get_innovation_number() < now_innovation:
                this_index += 1
            this_has_it = False
            if this_index < len(this_connections):
                this_has_it = this_connections[this_index].get_innovation_number() == now_innovation

            while other_index < len(other_connections) and \
                    other_connections[other_index].get_innovation_number() < now_innovation:
                other_index += 1
            other_has_it = False
            if other_index < len(other_connections):
                other_has_it = other_connections[other_index].get_innovation_number() == now_innovation

            if this_has_it != other_has_it:
                if this_index >= len(this_connections) or other_index >= len(other_connections):
                    exceed += 1
                else:
                    disjoint += 1

            if this_has_it and other_has_it:
                this_w = this_connections[this_index].weight
                other_w = other_connections[other_index].weight
                diff = math.fabs(this_w - other_w)
                w_diff += diff

        return ((exceed_penalty * exceed) / n) + ((disjoint_penalty * disjoint) / n) + w_diff_penalty * w_diff
