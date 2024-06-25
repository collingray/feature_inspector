from dataclasses import dataclass
import json
from typing import Dict

import torch


class ExamplesData:
    def __init__(self, device, dtype):
        self.seq_layer_pos_token = torch.tensor([], dtype=torch.int32, device=device)  # [num_examples, 4]
        self.activations = torch.tensor([], dtype=dtype, device=device)  # [num_examples]

    def sort(self):
        _, sort_indices = self.activations.sort(descending=True)
        self.seq_layer_pos_token = self.seq_layer_pos_token[sort_indices]
        self.activations = self.activations[sort_indices]

    def get_token_data(self):
        counts = torch.bincount(self.seq_layer_pos_token[:, 3])
        indices = torch.where(counts > 0)[0].unsqueeze(dim=1)
        avg_acts = (torch.bincount(self.seq_layer_pos_token[:, 3], weights=self.activations) / counts)

        return indices, counts[indices], avg_acts[indices]

    def get_layer(self, layer: int):
        mask = self.seq_layer_pos_token[:, 1] == layer
        return self.seq_layer_pos_token[mask], self.activations[mask]

    def save(self, dir: str, name: str):
        torch.save({
            'seq_layer_pos_token': self.seq_layer_pos_token,
            'activation': self.activations
        }, f"{dir}/{name}.pt")

    @classmethod
    def load(cls, dir: str, name, device, dtype):
        data = torch.load(f"{dir}/{name}.pt", map_location=device)
        examples = cls(device=device, dtype=dtype)
        examples.seq_layer_pos_token = data['seq_layer_pos_token']
        examples.activations = data['activation']
        return examples


@dataclass
class FeatureData:
    num: int
    examples: ExamplesData
    token_data: Dict[str, dict]

    def save(self, dir: str):
        with open(f"{dir}/{self.num}.json", 'w') as f:
            f.write(json.dumps({
                'num': self.num,
                'token_data': self.token_data
            }))

        self.examples.save(dir, str(self.num))

    @classmethod
    def load(cls, dir: str, name: str, device, dtype):
        with open(f"{dir}/${name}.json", 'r') as f:
            data = json.load(f)
            num = data['num']
            token_data = data['token_data']

        examples = ExamplesData.load(dir, name, device=device, dtype=dtype)

        return cls(num, examples, token_data)

    @classmethod
    def empty(cls, num: int, device: str, dtype: torch.dtype):
        return cls(num, ExamplesData(device=device, dtype=dtype), {})

    def add_examples(self, seq_layer_pos_token: torch.Tensor, activations: torch.Tensor):
        self.examples.seq_layer_pos_token = torch.cat((self.examples.seq_layer_pos_token, seq_layer_pos_token))
        self.examples.activations = torch.cat((self.examples.activations, activations))

    def record_token_data(self, decoder):
        tokens, counts, average_acts = self.examples.get_token_data()
        _, indices = counts[:, 0].sort(descending=True)
        tokens = tokens[indices]
        counts = counts[indices]
        average_acts = average_acts[indices]

        token_data = {}
        for i in range(tokens.size(0)):
            token = decoder(tokens[i])
            token_data[token] = {
                'count': counts[i].item(),
                'avg_activation': average_acts[i].item()
            }

        self.token_data = token_data
