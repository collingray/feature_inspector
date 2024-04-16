from typing import List, Union

from features import Feature
from display_utils import display_features
from inspector_widget import InspectorWidget


class Inspector:
    def __init__(self, features: List[Feature], num_features, num_layers):
        self.num_features = num_features
        self.num_layers = num_layers
        self.features = features

    @classmethod
    def index_features(cls):
        pass

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

        features = [self.features[feature] for feature in features]

        return display_features(features, layers, examples_per_layer)

    def display(self):
        return InspectorWidget(
            self.num_features,
            self.num_layers,
            self.display_features
        )
