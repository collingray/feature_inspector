import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterator, Tuple, Callable, Optional, Set

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from .features import FeatureData
from .inspector_widget import InspectorWidget


class Inspector:
    def __init__(
            self,
            feature_data: List[FeatureData],
            feature_occurrences: torch.Tensor,
            possible_occurrences: int,
            sequences: List[Tuple[str, List[int]]],
            bookmarked_features: Set[int],
            num_features: int,
            num_layers: int
    ):
        self.feature_data = feature_data
        self.feature_occurrences = feature_occurrences
        self.possible_occurrences = possible_occurrences
        self.sequences = sequences
        self.bookmarked_features = bookmarked_features
        self.num_features = num_features
        self.num_layers = num_layers

    @staticmethod
    def _tl_encoded_generator(
            model: HookedTransformer,
            dataset,
            encode_fn: Callable[[torch.Tensor], torch.Tensor],
            batch_size: int,
            act_site: str,
            dtype: torch.dtype,
            max_seq_length,
            num_layers,
    ):
        act_names = [f"blocks.{i}.{act_site}" for i in range(num_layers)]

        while True:
            for i in range(len(dataset) // batch_size):
                tokens = model.tokenizer(dataset[i:i+batch_size], max_length=max_seq_length, padding=True, return_tensors="pt")["input_ids"].to('cuda')

                output, cache = model.run_with_cache(tokens, names_filter=act_names)

                # [num_layers, batch_dim, seq_length, n_dim]
                mlp_outs = torch.stack([cache[act_name] for act_name in act_names])

                # [batch_dim, seq_length, num_layers, n_dim]
                mlp_outs = mlp_outs.permute(1, 2, 0, 3).to(dtype=dtype)

                # [batch_dim, seq_length, num_layers, m_dim]
                out = encode_fn(mlp_outs.flatten(0, 1)).view(mlp_outs.shape[:-1] + (-1,))

                yield tokens, out

    @classmethod
    def tl_index_features(
            cls,
            model: HookedTransformer,
            dataset,
            encode_fn: Callable[[torch.Tensor], torch.Tensor],
            num_features: int,
            num_layers: int,
            batch_size: int = 8,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            act_site: str = "hook_mlp_out",
            num_seqs: int = 4096,
            max_seq_length: Optional[int] = None,
            max_examples: int = 1024,
            activation_threshold: int = 1e-2
    ):
        encoded_generator = cls._tl_encoded_generator(
            model,
            dataset,
            encode_fn,
            batch_size,
            act_site,
            dtype,
            max_seq_length,
            num_layers
        )

        return cls.index_features(
            num_features,
            num_layers,
            encoded_generator,
            batch_size,
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
            batch_size: Optional[int],
            decode_tokens: Callable[[List[int]], str],
            device: str = "cuda",
            dtype=torch.bfloat16,
            num_seqs: int = 4096,
            max_examples: int = 128,
            activation_threshold: int = 1e-2
    ):
        initial_features = [
            FeatureData.empty(i, device=device, dtype=dtype)
            for i in range(num_features)
        ]
        feature_occurrences = torch.zeros(num_layers, num_features, dtype=torch.int, device=device)

        self = cls(initial_features, feature_occurrences, 0, [], set(), num_features, num_layers)

        feature_mask = torch.ones(num_layers, num_features, dtype=torch.bool, device=device)

        for _ in tqdm(range(num_seqs // (batch_size or 1))):
            tokens, out = encoded_generator.__next__()
            if batch_size is None:
                tokens = tokens.unsqueeze(0)  # [batch_size, seq_length]
                out = out.unsqueeze(0)  # [batch_size, seq_length, num_layers, m_dim]

            self.sequences += [self._decode_token_breaks(tokens[i], decode_tokens) for i in range(tokens.size(0))]

            self.possible_occurrences += out.size(0) * out.size(1) * num_layers

            activated_features = out > activation_threshold
            # [4, n] - feat, batch, pos, layer - permuted to have features as the first dim, ensuring they are contiguous
            example_act_indices = torch.stack(torch.where((activated_features * feature_mask).permute(3, 0, 1, 2)))

            self.feature_occurrences += activated_features.int().sum(dim=(0, 1))
            feature_mask[:] = self.feature_occurrences < max_examples

            if example_act_indices.size(1) == 0:
                continue

            block_idxs = torch.cat((
                torch.tensor([0], device=device),
                torch.where(example_act_indices[0, 1:] != example_act_indices[0, :-1])[0] + 1,
                torch.tensor([example_act_indices.size(1)], device=device)
            ))

            features: torch.Tensor = example_act_indices[0, block_idxs[:-1]]

            blocks = []
            for i in range(len(block_idxs) - 1):
                # [3, block_size] - seq, pos, layer
                block = example_act_indices[1:, block_idxs[i]:block_idxs[i + 1]]

                # [4, block_size] - seq, pos, layer, token
                block = torch.cat((
                    block,
                    tokens[block[0], block[1]].unsqueeze(0)
                ))

                # [block_size, 4] - seq, layer, pos, token
                block = block[[0, 2, 1, 3]].swapdims(0, 1)

                blocks.append(block)

            # [example_act_indices[1:, block_idxs[i]:block_idxs[i + 1]] for i in range(len(block_idxs) - 1)]

            for i, feat in enumerate(features):
                block = blocks[i]
                # seq_layer_pos_token = torch.stack((  # [block_size, 4] - seq, layer, pos, token
                #     torch.full((block.size(1),), len(self.sequences) - 1, dtype=torch.int, device=device),
                #     block[1],
                #     block[0],
                #     tokens[block[0]]
                # )).swapdims(0, 1)
                activations = out[block[:, 0], block[:, 2], block[:, 1], feat]

                self.feature_data[feat].add_examples(block, activations)

        with ThreadPoolExecutor() as executor:
            executor.map(lambda f: (f.record_token_data(decode_tokens), f.examples.sort()), self.feature_data)

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

    def display(self):
        return InspectorWidget(
            inspector=self,
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
                "bookmarked_features": self.bookmarked_features,
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
        bookmarked_features = cfg_dict["bookmarked_features"]
        num_features = cfg_dict["num_features"]
        num_layers = cfg_dict["num_layers"]

        feature_data = []
        for i in range(num_features):
            try:
                feature_data.append(FeatureData.load(f"{path}/{name}", str(i), device=device, dtype=dtype))
            except FileNotFoundError:
                feature_data.append(FeatureData.empty(i, device=device, dtype=dtype))

        return cls(feature_data, feature_occurrences, possible_occurrences, sequences, set(bookmarked_features),
                   num_features, num_layers)
