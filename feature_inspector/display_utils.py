import math
from typing import List, Tuple

from ipywidgets import widgets

from .features import FeatureData, FeatureExample

FEATURE_CSS = """
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
    offset = 0

    for start, end, activation in subseqs:
        gb_value = 255 - max(min(math.log10(activation + 1) * 255, 255), 10)
        color = f"rgb(255, {gb_value}, {gb_value})"
        tooltip = f"<span class='feature-info'>{activation:.3f}</span>"
        token = f"<span class='feature' style='background-color: {color}'>{seq[start + offset:end + offset]}{tooltip}</span>"
        seq = seq[:start + offset] + token + seq[end + offset:]
        offset += len(token) - (end - start)

    return seq


def display_examples(examples: List[FeatureExample], seqs, examples_per_layer, context_width=10):
    chosen_examples: List[List[FeatureExample]] = []

    for example in examples:
        if len(chosen_examples) >= examples_per_layer:
            break

        for i in range(len(chosen_examples)):
            if chosen_examples[i][0].seq_num == example.seq_num:
                chosen_examples[i].append(example)
                break
        else:
            chosen_examples.append([example])

    out = []
    for example_group in chosen_examples:
        seq, token_breaks = seqs[example_group[0].seq_num][0]
        token_subseqs = [(token_breaks[example.seq_num], token_breaks[example.seq_num + 1], example.activation) for
                         example in example_group]
        seq = highlight_subseqs(seq, token_subseqs)
        start = max(0, min(token_subseqs, key=lambda x: x[0])[0] - context_width)
        end = min(len(seq), max(token_subseqs, key=lambda x: x[1])[1] + context_width)
        out.append(widgets.HTML(seq[start:end]))

    return widgets.VBox(out)


def display_feature(feature: FeatureData, layers: List[int], seqs: List[Tuple[str, List[int]]], examples_per_layer):
    children = [
        display_examples(list(feature.examples[layer]), seqs, examples_per_layer)
        for layer in layers
    ]
    accordion = widgets.Accordion(children=children, selected_index=0)

    for i in layers:
        accordion.set_title(i, f"Layer {i}")

    return accordion


def display_features(features: List[FeatureData], layers: List[int], seqs: List[Tuple[str, List[int]]],
                     examples_per_layer=3):
    if len(features) > 20:
        sections = min(10, len(features) // 10)
        section_width = math.ceil(len(features) / sections)

        children = [
            display_features(features[i:i + section_width], layers, seqs, examples_per_layer)
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
            display_feature(feature, layers, seqs, examples_per_layer)
            for feature in features
        ]

        tabs = widgets.Tab(children=children)

        for i, feature in enumerate(features):
            if len(features) <= 10:
                tabs.set_title(i, f"Feature {feature.num}")
            else:
                tabs.set_title(i, str(feature.num))

    # CSS for feature hover tooltips
    style = widgets.HTML(f"<style>{FEATURE_CSS}</style>")

    return widgets.VBox([style, tabs])
