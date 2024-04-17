from typing import List, Union, Iterator, Tuple, Callable

import torch
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from features import Feature, FeatureExample
from display_utils import display_features
from inspector_widget import InspectorWidget


class Inspector:
    def __init__(
            self,
            feature_examples: List[Feature],
            feature_occurrences: torch.Tensor,
            possible_occurrences: int,
            num_features: int,
            num_layers: int
    ):
        self.feature_examples = feature_examples
        self.feature_occurrences = feature_occurrences
        self.possible_occurrences = possible_occurrences
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

                token_strings = model.tokenizer.batch_decode(tokens)

                output, cache = model.run_with_cache(tokens, names_filter=act_names)

                # [num_layers, batch_dim, seq_length, n_dim]
                mlp_outs = torch.stack([cache[act_name] for act_name in act_names])

                # [seq_length, num_layers, n_dim]
                mlp_outs = mlp_outs.squeeze(dim=1).permute(1, 0, 2).to(dtype=dtype)

                # [seq_length, num_layers, m_dim]
                out = encode_fn(mlp_outs)

                yield out, token_strings

    @classmethod
    def tl_index_features(
            cls,
            model: HookedTransformer,
            dataset,
            encode_fn: Callable[[torch.Tensor], torch.Tensor],
            num_features,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            act_site: str = "hook_mlp_out",
            num_seqs: int = 4096,
            max_seq_length: int = 1024,
            min_seq_length: int = 16,
            max_examples: int = 512,
            context_width: int = 10,
            num_layers: int = 12,
            activation_threshold: int = 1e-3
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
            encoded_generator,
            num_features,
            device,
            num_seqs,
            max_examples,
            context_width,
            num_layers,
            activation_threshold
        )

    @classmethod
    def index_features(
            cls,
            encoded_generator: Iterator[Tuple[torch.Tensor, List[str]]],
            num_features,
            device: str = "cuda",
            num_seqs: int = 4096,
            max_examples: int = 512,
            context_width: int = 10,
            num_layers: int = 12,
            activation_threshold: int = 1e-3
    ):
        features = [
            Feature.empty(i, num_layers)
            for i in range(num_features)
        ]
        feature_occurrences = torch.zeros(num_layers, num_features, dtype=torch.int, device=device)

        self = cls(features, feature_occurrences, 0, num_features, num_layers)

        feature_mask = torch.ones(num_features, dtype=torch.bool, device=device)

        for _ in tqdm(range(num_seqs)):
            # [seq_length, num_layers, m_dim], [seq_length]
            out, token_strings = encoded_generator.__next__()

            self.possible_occurrences += out.size(0) * num_layers

            activated_features = out > activation_threshold
            example_act_indices = torch.where(activated_features * feature_mask)

            self.feature_occurrences += activated_features.int().sum(dim=0)
            feature_mask[:] = self.feature_occurrences.sum(dim=0) < max_examples

            prev_seq = -1
            context = ""
            for seq, layer, feat in zip(*example_act_indices):
                if self.feature_examples[feat].total_examples > max_examples:
                    continue

                if seq != prev_seq:  # if seq hasn't incremented, we don't need to recalculate the context
                    prev_seq = seq
                    start = max(0, seq - context_width)
                    end = seq + context_width + 1
                    idx = min(seq, context_width)
                    context_tokens = token_strings[start:end]
                    context_tokens[idx] = f"||{context_tokens[idx]}||"
                    context = "".join(context_tokens)

                self.feature_examples[feat].add_example(layer.item(), FeatureExample(out[seq, layer, feat].item(), context))

        self.feature_occurrences = self.feature_occurrences.cpu()

        return self

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

        return display_features(features, layers, examples_per_layer)

    def display(self):
        return InspectorWidget(
            self.num_features,
            self.num_layers,
            self.feature_occurrences,
            self.possible_occurrences,
            self.display_features
        )
