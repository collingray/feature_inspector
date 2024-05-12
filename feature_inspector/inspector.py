import json
import os
from typing import List, Union, Iterator, Tuple, Callable, Optional

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from .features import FeatureData, FeatureExample
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
        self.feature_examples = feature_data
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

                tokens = model.tokenizer(dataset[i], max_length=max_seq_length, return_tensors="pt")["input_ids"][0]

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
            num_seqs,
            max_examples,
            activation_threshold
        )

    @classmethod
    def index_features(
            cls,
            num_features: int,
            num_layers: int,
            encoded_generator: Iterator[Tuple[List[int], torch.Tensor]],
            decode_tokens: Callable[[List[int]], str],
            device: str = "cuda",
            num_seqs: int = 4096,
            max_examples: int = 128,
            activation_threshold: int = 1e-2
    ):
        features = [
            FeatureData.empty(i, num_layers)
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
            example_act_indices = torch.where(activated_features * feature_mask)

            self.feature_occurrences += activated_features.int().sum(dim=0)
            feature_mask[:] = self.feature_occurrences < max_examples

            for pos, layer, feat in zip(*example_act_indices):
                if len(self.feature_examples[feat].examples[layer]) <= max_examples:
                    self.feature_examples[feat].add_example(
                        layer,
                        FeatureExample(
                            out[pos, layer, feat].item(),
                            len(self.sequences) - 1,
                            tokens[pos],
                            decode_tokens([tokens[pos]])
                        )
                    )

        self.feature_occurrences = self.feature_occurrences.cpu()

        return self

    @staticmethod
    def _decode_token_breaks(tokens, decode_tokens):
        """
        Decodes the tokens and returns the sequence and the indices of the token breaks.

        Needed as some tokens may not be a valid utf-8 string, so we can't just join them.
        """
        token_breaks = []
        seq = ""
        token_breaks = [0]
        token_acc = []

        for token in tokens:
            token_acc.append(token)
            decoded = decode_tokens(token_acc)
            if ord(decoded) != 65533:
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

        features = [self.feature_examples[feature] for feature in features]

        return display_features(features, layers, self.sequences, examples_per_layer)

    def display(self):
        return InspectorWidget(
            self.num_features,
            self.num_layers,
            self.feature_occurrences,
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

        for feature in self.feature_examples:
            with open(f"{path}/{name}/{feature.num}.json", "w") as f:
                f.write(feature.to_json())

    @classmethod
    def load(cls, path, name):
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

        features_examples = []
        for i in range(num_features):
            try:
                with open(f"{path}/{name}/{i}.json", "r") as f:
                    features_examples.append(FeatureData.from_json(f.read()))
            except FileNotFoundError:
                features_examples.append(FeatureData.empty(i, num_layers))

        return cls(features_examples, feature_occurrences, possible_occurrences, sequences, num_features, num_layers)
