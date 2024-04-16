from typing import Callable

import ipywidgets as widgets

from widget_controls import FeatureControls, LayerControls, GeneralControls


class InspectorWidget(widgets.VBox):
    def __init__(self, num_features: int, num_layers: int, display_fn: Callable[[list, list, int], widgets.Widget]):
        self.num_features = num_features
        self.num_layers = num_layers
        self.display_fn = display_fn

        self.feature_controls = FeatureControls(num_features)
        self.layer_controls = LayerControls(num_layers)
        self.general_controls = GeneralControls()
        self.display = self.display_fn(
            self.feature_controls.features,
            self.layer_controls.layers,
            self.general_controls.examples_per_layer.value
        )

        self.feature_controls.layout.border = "1px solid black"
        self.layer_controls.layout.border_left = "1px solid black"
        self.layer_controls.layout.border_right = "1px solid black"
        self.general_controls.layout.border = "1px solid black"

        self.feature_controls.layout.padding = "10px"
        self.layer_controls.layout.padding = "10px"
        self.general_controls.layout.padding = "10px"

        children = [
            self.feature_controls,
            self.layer_controls,
            self.general_controls,
            self.display
        ]

        self.general_controls.render_button.on_click(lambda _: self.redraw_display())

        super().__init__(children=children)

    def redraw_display(self):
        self.display.close()
        self.display = self.display_fn(
            self.feature_controls.features,
            self.layer_controls.layers,
            self.general_controls.examples_per_layer.value
        )

        self.children = [
            self.feature_controls,
            self.layer_controls,
            self.general_controls,
            self.display
        ]
