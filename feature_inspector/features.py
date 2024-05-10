from dataclasses import dataclass
import json
from typing import List


@dataclass
class FeatureExample:
    activation: float
    context: str
    tok_start: int
    tok_end: int

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__)

    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))


@dataclass
class Feature:
    num: int
    layers: List[List[FeatureExample]]
    total_examples: int = 0

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        layers = [
            [FeatureExample(**example) for example in layer]
            for layer in data['layers']
        ]

        return cls(data['num'], layers, data['total_examples'])

    @classmethod
    def empty(cls, num: int, layers: int):
        return cls(num, [[] for _ in range(layers)])

    def add_example(self, layer: int, example: FeatureExample):
        self.layers[layer].append(example)
        self.layers[layer].sort(key=lambda x: x.activation, reverse=True)
        self.total_examples += 1
