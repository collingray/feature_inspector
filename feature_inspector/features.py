from dataclasses import dataclass
import json
from typing import Dict

import torch


class ExamplesData:
    def __init__(self):
        self.seq_layer_pos_token = torch.tensor([], dtype=torch.long)  # [num_examples, 4]
        self.activations = torch.tensor([], dtype=torch.float)  # [num_examples]

    def sort(self):
        _, sort_indices = self.activations.sort(descending=True)
        self.seq_layer_pos_token = self.seq_layer_pos_token[sort_indices]
        self.activations = self.activations[sort_indices]

    def get_token_data(self):
        counts = torch.bincount(self.seq_layer_pos_token[:, 3])
        indices = torch.where(counts > 0)[0].unsqueeze(dim=1)
        avg_acts = (torch.bincount(self.seq_layer_pos_token[:, 3], weights=self.activations) / counts)

        return torch.cat((indices, counts[indices], avg_acts[indices]), dim=-1)

    def get_layer(self, layer: int):
        mask = self.seq_layer_pos_token[:, 1] == layer
        return self.seq_layer_pos_token[mask], self.activations[mask]

    def save(self, dir: str, name: str):
        torch.save({
            'seq_layer_pos_token': self.seq_layer_pos_token,
            'activation': self.activations
        }, f"{dir}/{name}.pt")

    @classmethod
    def load(cls, dir: str, name):
        data = torch.load(f"{dir}/{name}.pt")
        examples = cls()
        examples.seq_layer_pos_token = data['seq_layer_pos_token']
        examples.activations = data['activation']
        return examples


@dataclass
class FeatureData:
    num: int
    examples: ExamplesData
    token_data: Dict[str, dict]

    def save(self, dir: str):
        with open(f"{dir}/${self.num}.json", 'w') as f:
            f.write(json.dumps({
                'num': self.num,
                'token_counts': self.token_data
            }))

        self.examples.save(dir, str(self.num))

    @classmethod
    def load(cls, dir: str, name: str):
        with open(f"{dir}/${name}.json", 'r') as f:
            data = json.load(f)
            num = data['num']
            token_data = data['token_counts']

        examples = ExamplesData.load(dir, name)

        return cls(num, examples, token_data)

    @classmethod
    def empty(cls, num: int, layers: int):
        return cls(num, ExamplesData(), {})

    def add_examples(self, seq_layer_pos_token: torch.Tensor, activations: torch.Tensor):
        self.examples.seq_layer_pos_token = torch.cat((self.examples.seq_layer_pos_token, seq_layer_pos_token))
        self.examples.activations = torch.cat((self.examples.activations, activations))

    def record_token_data(self, decoder):
        data = self.examples.get_token_data()
        token_data = {}
        for i in range(data.size(0)):
            token = decoder.decode(data[i, 0].item())
            token_data[token] = {
                'count': data[i, 1].item(),
                'avg_activation': data[i, 2].item()
            }

        self.token_data = token_data
