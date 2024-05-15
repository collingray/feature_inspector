import math
from typing import List, Tuple, Callable, Set

import torch
from ipywidgets import widgets
import html

from .features import FeatureData, ExamplesData

FEATURE_CSS = """
p {
    line-height: 1.3;
}

.feature {
  position: relative;
  border-bottom: 1px dotted black;
}

.feature .feature-info {
  visibility: hidden;
  background-color: white;
  color: black;
  text-align: left;
  padding: 5px;
  border: 1px black solid;
  border-radius: 2px;
  overflow: show;
  min-width: 100%;
  top: 100%;
  left: 0;
  white-space: nowrap;
  position: absolute;
  z-index: 1;
}

.feature:hover .feature-info {
  visibility: visible;
}"""


def highlight_subseqs(seq: str, subseqs: List[Tuple[int, int, float]]):
    subseqs.sort(key=lambda x: x[0])

    out = ""

    for i, (start, end, activation) in enumerate(subseqs):
        gb_value = 255 - max(min(math.log10(activation + 1) * 255, 255), 10)
        color = f"rgb(255, {gb_value}, {gb_value})"
        tooltip = f"<span class='feature-info'>{activation:.3f}</span>"
        token = f"<span class='feature' style='background-color: {color}'>{html.escape(seq[start:end])}{tooltip}</span>"

        if i == 0:
            out += html.escape(seq[:start])
        else:
            out += html.escape(seq[subseqs[i - 1][1]:start])

        out += token

        if i == len(subseqs) - 1:
            out += html.escape(seq[end:])

    return out


def display_examples(examples, seqs, examples_per_layer, context_width=50):
    seq_layer_pos_token = examples[0]
    activations = examples[1]

    chosen_seqs = set()
    for i in range(len(seq_layer_pos_token)):
        chosen_seqs.add(seq_layer_pos_token[i][0].item())
        if len(chosen_seqs) >= examples_per_layer:
            break
    chosen_seqs = torch.tensor(list(chosen_seqs))
    chosen_mask = torch.isin(seq_layer_pos_token[:, 0], chosen_seqs)
    chosen_examples = seq_layer_pos_token[chosen_mask]
    chosen_example_activations = activations[chosen_mask]

    out = []
    for seq_num in chosen_seqs:
        seq, token_breaks = seqs[seq_num]
        token_subseqs = []
        for i in torch.where(chosen_examples[:, 0] == seq_num)[0]:
            seq_num, _, pos, _ = chosen_examples[i]
            act = chosen_example_activations[i]
            token_subseqs.append((token_breaks[pos], token_breaks[pos + 1], act))

        start = max(0, min(token_subseqs, key=lambda x: x[0])[0] - context_width)
        end = min(len(seq), max(token_subseqs, key=lambda x: x[1])[1] + context_width)
        seq = seq[:end]

        seq = highlight_subseqs(seq, token_subseqs).replace("\n", "⏎")
        out.append(widgets.HTML(seq[start:]))
        out.append(widgets.HTML(f"<hr>"))

    return widgets.VBox(out)


def feature_controls(feature: FeatureData, bookmark_feature_fn: Callable[[int], None]):
    bookmark_button = widgets.Button(
        description='Bookmark',
        disabled=False,
        tooltip='Bookmark feature',
        icon='star'
    )

    bookmark_button.style.button_color = 'gold'
    bookmark_button.on_click(lambda _: bookmark_feature_fn(feature.num))

    labels = [
        widgets.Label(f"Average activation: {feature.examples.activations.mean().item():.3f}"),
        widgets.Label(f"Unique tokens: {len(feature.token_data)}"),
        widgets.Label(f"Top tokens: [ {' | '.join(list(feature.token_data.keys())[:10])} ]")
    ]

    for label in labels:
        label.layout.width = "auto"

    return widgets.HBox(
        [bookmark_button] + labels
    )


def display_feature(feature: FeatureData, layers: List[int], seqs: List[Tuple[str, List[int]]],
                    bookmark_feature_fn: Callable[[int], None], examples_per_layer):
    children = [
        display_examples(feature.examples.get_layer(layer), seqs, examples_per_layer)
        for layer in layers
    ]
    accordion = widgets.Accordion(children=children, selected_index=0)

    for i in layers:
        average_activation = feature.examples.get_layer(i)[1].mean().item()
        accordion.set_title(i, f"Layer {i} - (avg act: {average_activation:.3f})")

    return widgets.VBox([
        feature_controls(feature, bookmark_feature_fn),
        accordion
    ])


def display_features(
        features: List[FeatureData],
        layers: List[int],
        seqs: List[Tuple[str, List[int]]],
        bookmark_feature_fn: Callable[[int], None],
        examples_per_layer=3
):
    if len(features) > 20:
        sections = min(10, len(features) // 10)
        section_width = math.ceil(len(features) / sections)

        children = [
            display_features(features[i:i + section_width], layers, seqs, bookmark_feature_fn, examples_per_layer)
            for i in range(0, len(features), section_width)
        ]

        tabs = widgets.Tab(children=children)

        for i in range(0, len(features), section_width):
            tabs.set_title(
                i // section_width,
                f"{features[i].num}-{features[min(i + section_width, len(features)) - 1].num}"
            )

    else:
        children = [
            display_feature(feature, layers, seqs, bookmark_feature_fn, examples_per_layer)
            for feature in features
        ]

        tabs = widgets.Tab(children=children)

        for i, feature in enumerate(features):
            if len(features) <= 8:
                tabs.set_title(i, f"Feature {feature.num}")
            else:
                tabs.set_title(i, str(feature.num))

    # CSS for feature hover tooltips
    style = widgets.HTML(f"<style>{FEATURE_CSS}</style>")

    return widgets.VBox([style, tabs])
