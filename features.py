from dataclasses import dataclass
from typing import List


@dataclass
class FeatureExample:
    activation: float
    context: str


@dataclass
class Feature:
    num: int
    layers: List[List[FeatureExample]]
    total_examples: int = 0

    @classmethod
    def empty(cls, num: int, layers: int):
        return cls(num, [[] for _ in range(layers)])

    def add_example(self, layer: int, example: FeatureExample):
        self.layers[layer].append(example)
        self.total_examples += 1
