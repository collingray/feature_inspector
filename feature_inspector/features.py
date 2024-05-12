from dataclasses import dataclass
import json
from typing import List, Dict

from sortedcontainers import SortedList


@dataclass
class FeatureExample:
    activation: float
    seq_num: int
    token: int
    token_str: str

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__)

    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))


@dataclass
class FeatureData:
    num: int
    examples: List[SortedList[FeatureExample]]
    total_examples: int
    token_data: Dict[str, dict]

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        examples = [
            SortedList(iterable=[FeatureExample(**example) for example in layer], key=lambda x: -x.activation)
            for layer in data['examples']
        ]

        return cls(data['num'], examples, data['total_examples'], data['token_counts'])

    @classmethod
    def empty(cls, num: int, layers: int):
        return cls(num, [SortedList[FeatureExample](key=lambda x: -x.activation) for _ in range(layers)], 0, {})

    def add_example(self, layer: int, example: FeatureExample):
        self.examples[layer].add(example)
        self.total_examples += 1

        if self.token_data[example.token_str] is None:
            self.token_data[example.token_str] = {
                'count': 1,
                'avg_activation': example.activation
            }
        else:
            curr_data = self.token_data[example.token_str]
            self.token_data[example.token_str]['avg_activation'] = (
                    (curr_data['avg_activation'] * curr_data['count'] + example.activation) / (curr_data['count'] + 1)
            )
            self.token_data[example.token_str]['count'] += 1
