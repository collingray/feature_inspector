import math
from typing import List

from ipywidgets import widgets

from features import Feature, FeatureExample


def display_example(example: FeatureExample):
    return widgets.Label(f"{example.activation} - {example.context}")


def display_feature(feature: Feature, layers: List[int], examples_per_layer=3):
    children = [
        widgets.VBox([display_example(example) for example in feature.layers[layer][:examples_per_layer]])
        for layer in layers
    ]
    accordion = widgets.Accordion(children=children, selected_index=0)

    for i in layers:
        accordion.set_title(i, f"Layer {i}")

    return accordion


def display_features(features: List[Feature], layers: List[int], examples_per_layer=3):
    if len(features) > 20:
        sections = min(10, len(features) // 10)
        section_width = math.ceil(len(features) / sections)

        children = [
            display_features(features[i:i + section_width], layers, examples_per_layer)
            for i in range(0, len(features), section_width)
        ]

        tabs = widgets.Tab(children=children)

        for i in range(0, len(features), section_width):
            tabs.set_title(i // section_width, f"{i}-{min(i + section_width, len(features))-1}")

    else:
        children = [
            display_feature(feature, layers, examples_per_layer)
            for feature in features
        ]

        tabs = widgets.Tab(children=children)

        for i, feature in enumerate(features):
            if len(features) <= 10:
                tabs.set_title(i, f"Feature {feature.num}")
            else:
                tabs.set_title(i, str(feature.num))

    return tabs
