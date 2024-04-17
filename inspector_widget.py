from typing import Callable

import ipywidgets as widgets
import torch

from graph_widgets import FeatureFrequencyRange, LayersActivatedRange
from widget_controls import FeatureControls, LayerControls, GeneralControls


class InspectorWidget(widgets.VBox):
    def __init__(
            self,
            num_features: int,
            num_layers: int,
            feature_occurrences: torch.Tensor,
            possible_occurrences: int,
            display_fn: Callable[[list, list, int], widgets.Widget]
    ):
        self.num_features = num_features
        self.num_layers = num_layers
        self.display_fn = display_fn

        self.feature_controls = FeatureControls(num_features)
        self.layer_controls = LayerControls(num_layers)
        self.general_controls = GeneralControls()
        self.frequency_histogram = FeatureFrequencyRange(feature_occurrences, possible_occurrences)
        self.layers_histogram = LayersActivatedRange(feature_occurrences)
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
            self.display,
        ]

        self.general_controls.render_button.on_click(lambda _: self.redraw_display())
        self.general_controls.enable_graph_filters.observe(lambda state: self.refresh())

        super().__init__(children=children)

    def redraw_display(self):
        self.general_controls.render_button.disabled = True
        self.general_controls.render_button.description = "Rendering..."
        self.general_controls.render_button.button_style = "info"
        self.general_controls.render_button.icon = "gear spin lg"

        try:
            self.display.close()
            self.display = self.display_fn(
                self.feature_controls.features,
                self.layer_controls.layers,
                self.general_controls.examples_per_layer.value
            )

            self.refresh()

            self.general_controls.render_button.button_style = "success"
            self.general_controls.render_button.icon = "check"
        except:
            self.general_controls.render_button.button_style = "danger"
            self.general_controls.render_button.icon = "exclamation-triangle"

        self.general_controls.render_button.disabled = False
        self.general_controls.render_button.description = "Render"

    def refresh(self):
        if self.general_controls.enable_graph_filters.value:
            self.children = [
                self.feature_controls,
                self.layer_controls,
                self.general_controls,
                self.frequency_histogram,
                self.layers_histogram,
                self.display,
            ]
        else:
            self.children = [
                self.feature_controls,
                self.layer_controls,
                self.general_controls,
                self.display,
            ]
