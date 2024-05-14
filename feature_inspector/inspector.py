import json
import os
from multiprocessing import Pool
from typing import List, Union, Iterator, Tuple, Callable, Optional

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from .features import FeatureData
from .display_utils import display_features
from .inspector_widget import InspectorWidget


class Inspector:
    def __init__(
            self,
            feature_data: List[FeatureData],
            feature_occurrences: torch.Tensor,
            possible_occurrences: int,
            sequences: List[Tuple[str, List[int]]],
            num_features: int,
            num_layers: int
    ):
        self.feature_data = feature_data
        self.feature_occurrences = feature_occurrences
        self.possible_occurrences = possible_occurrences
        self.sequences = sequences
        self.num_features = num_features
        self.num_layers = num_layers

    @staticmethod
    def _tl_encoded_generator(
            model: HookedTransformer,
            dataset,
            encode_fn: Callable[[torch.Tensor], torch.Tensor],
            act_site: str,
            dtype: torch.dtype,
            max_seq_length,
            min_seq_length,
            num_layers,
    ):
        while True:
            for i in range(len(dataset)):
                act_names = [f"blocks.{i}.{act_site}" for i in range(num_layers)]

                tokens = model.tokenizer(dataset[i], max_length=max_seq_length, return_tensors="pt")["input_ids"][0].to(
                    'cuda')

                if len(tokens) < min_seq_length:
                    continue

                output, cache = model.run_with_cache(tokens, names_filter=act_names)

                # [num_layers, batch_dim, seq_length, n_dim]
                mlp_outs = torch.stack([cache[act_name] for act_name in act_names])

                # [seq_length, num_layers, n_dim]
                mlp_outs = mlp_outs.squeeze(dim=1).permute(1, 0, 2).to(dtype=dtype)

                # [seq_length, num_layers, m_dim]
                out = encode_fn(mlp_outs)

                yield tokens, out

    @classmethod
    def tl_index_features(
            cls,
            model: HookedTransformer,
            dataset,
            encode_fn: Callable[[torch.Tensor], torch.Tensor],
            num_features: int,
            num_layers: int,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            act_site: str = "hook_mlp_out",
            num_seqs: int = 4096,
            max_seq_length: Optional[int] = None,
            min_seq_length: int = 16,
            max_examples: int = 1024,
            activation_threshold: int = 1e-2
    ):
        encoded_generator = cls._tl_encoded_generator(
            model,
            dataset,
            encode_fn,
            act_site,
            dtype,
            max_seq_length,
            min_seq_length,
            num_layers
        )

        return cls.index_features(
            num_features,
            num_layers,
            encoded_generator,
            model.tokenizer.decode,
            device,
            dtype,
            num_seqs,
            max_examples,
            activation_threshold
        )

    @classmethod
    def index_features(
            cls,
            num_features: int,
            num_layers: int,
            encoded_generator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
            decode_tokens: Callable[[List[int]], str],
            device: str = "cuda",
            dtype=torch.bfloat16,
            num_seqs: int = 4096,
            max_examples: int = 128,
            activation_threshold: int = 1e-2
    ):
        features = [
            FeatureData.empty(i, device=device, dtype=dtype)
            for i in range(num_features)
        ]
        feature_occurrences = torch.zeros(num_layers, num_features, dtype=torch.int, device=device)

        self = cls(features, feature_occurrences, 0, [], num_features, num_layers)

        feature_mask = torch.ones(num_layers, num_features, dtype=torch.bool, device=device)

        for _ in tqdm(range(num_seqs)):
            # [seq_length, num_layers, m_dim], [seq_length]
            tokens, out = encoded_generator.__next__()

            seq, token_breaks = self._decode_token_breaks(tokens, decode_tokens)

            self.sequences.append((seq, token_breaks))

            self.possible_occurrences += out.size(0) * num_layers

            activated_features = out > activation_threshold
            # permute to have features as the first dim, ensuring they are contiguous
            example_act_indices = torch.stack(torch.where((activated_features * feature_mask).permute(2, 0, 1)))

            self.feature_occurrences += activated_features.int().sum(dim=0)
            feature_mask[:] = self.feature_occurrences < max_examples

            if example_act_indices.size(1) == 0:
                continue

            feature_blocks = torch.cat((
                torch.tensor([0], device=device),
                torch.where(example_act_indices[0][1:] != example_act_indices[0][:-1])[0] + 1,
                torch.tensor([len(example_act_indices[0])], device=device)
            ))

            features = example_act_indices[0, feature_blocks[:-1]]

            for i, feat in enumerate(features):
                block_start = feature_blocks[i]
                block_end = feature_blocks[i + 1]
                feat = example_act_indices[0, block_start]

                block = example_act_indices[1:, block_start:block_end]  # [2, block_size] - pos, layer
                seq_layer_pos_token = torch.stack((  # [block_size, 4] - seq, layer, pos, token
                    torch.full((block.size(1),), len(self.sequences) - 1, dtype=torch.int, device=device),
                    block[1],
                    block[0],
                    tokens[block[0]]
                )).swapdims(0, 1)
                activations = out[block[0], block[1], feat]

                self.feature_data[feat].add_examples(seq_layer_pos_token, activations)

        for i in range(len(self.feature_data)):
            self.feature_data[i].record_token_data(decode_tokens)
            self.feature_data[i].examples.sort()

        return self

    @staticmethod
    def _decode_token_breaks(tokens, decode_tokens):
        """
        Decodes the tokens and returns the sequence and the indices of the token breaks.

        Needed as some tokens may not be a valid utf-8 string, so we can't just join them.
        """
        seq = ""
        token_breaks = [0]
        token_acc = []

        for token in tokens:
            token_acc.append(token)
            decoded = decode_tokens(token_acc)
            if '�' not in decoded:
                seq += decoded
                token_acc = []

            token_breaks.append(len(seq))

        return seq, token_breaks

    def display_features(
            self,
            features: Union[int, range, list],
            layers: Union[int, range, list],
            examples_per_layer=3
    ):
        if isinstance(features, int):
            features = [features]
        else:
            features = list(features)

        if isinstance(layers, int):
            layers = [layers]
        else:
            layers = list(layers)

        features = [self.feature_data[feature] for feature in features]

        return display_features(features, layers, self.sequences, examples_per_layer)

    def display(self):
        return InspectorWidget(
            self.num_features,
            self.num_layers,
            self.feature_occurrences,
            torch.stack([feature.examples.activations.mean() for feature in self.feature_data]).detach().to(
                dtype=torch.float32),
            self.possible_occurrences,
            self.display_features
        )

    def save(self, path, name):
        """
        Saves the inspector config to a json file, and saves all the features to individual json files
        :param path: The directory to save the files to
        :param name: The name (without extension) to use for the inspector config and directory of the features
        """
        with open(f"{path}/{name}.cfg", "w") as f:
            cfg_dict = {
                "feature_occurrences": self.feature_occurrences.tolist(),
                "possible_occurrences": self.possible_occurrences,
                "sequences": self.sequences,
                "num_features": self.num_features,
                "num_layers": self.num_layers,
            }

            json.dump(cfg_dict, f)

        # ensure the directory exists
        os.makedirs(f"{path}/{name}", exist_ok=True)

        for feature in self.feature_data:
            feature.save(f"{path}/{name}")

    @classmethod
    def load(cls, path, name, device, dtype):
        """
        Loads an inspector config and features from a directory
        :param path: The directory containing the files
        :param name: The name (without extension) of the inspector config and directory of the features
        :return: An Inspector instance
        """
        with open(f"{path}/{name}.cfg", "r") as f:
            cfg_dict = json.load(f)

        feature_occurrences = torch.tensor(cfg_dict["feature_occurrences"])
        possible_occurrences = cfg_dict["possible_occurrences"]
        sequences = cfg_dict["sequences"]
        num_features = cfg_dict["num_features"]
        num_layers = cfg_dict["num_layers"]

        feature_data = []
        for i in range(num_features):
            try:
                feature_data.append(FeatureData.load(f"{path}/{name}", str(i), device=device, dtype=dtype))
            except FileNotFoundError:
                feature_data.append(FeatureData.empty(i, device=device, dtype=dtype))

        return cls(feature_data, feature_occurrences, possible_occurrences, sequences, num_features, num_layers)
