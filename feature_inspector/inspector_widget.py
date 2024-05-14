import traceback
from typing import Callable

import ipywidgets as widgets
import torch

from .filter_widget import FilterWidget
from .control_widgets import FeatureControls, LayerControls, GeneralControls


class InspectorWidget(widgets.VBox):
    def __init__(
            self,
            num_features: int,
            num_layers: int,
            feature_occurrences: torch.Tensor,
            average_activations: torch.Tensor,
            possible_occurrences: int,
            display_fn: Callable[[list, list, int], widgets.Widget]
    ):
        self.num_features = num_features
        self.num_layers = num_layers
        self.display_fn = display_fn

        self.feature_controls = FeatureControls(num_features)
        self.layer_controls = LayerControls(num_layers)
        self.general_controls = GeneralControls()
        self.filter_widget = FilterWidget(feature_occurrences, average_activations, possible_occurrences)
        self.display = self.display_fn(
            self.feature_controls.features,
            self.layer_controls.layers,
            self.general_controls.examples_per_layer.value
        )

        self.feature_controls.layout.border = "1px solid black"
        self.feature_controls.layout.padding = "10px"

        for widget in [self.layer_controls, self.general_controls, self.filter_widget]:
            widget.layout.border_left = "1px solid black"
            widget.layout.border_right = "1px solid black"
            widget.layout.border_bottom = "1px solid black"
            widget.layout.padding = "10px"

        children = [
            self.feature_controls,
            self.layer_controls,
            self.general_controls,
            self.display,
        ]

        self.general_controls.render_button.on_click(lambda _: self.redraw_display())
        self.general_controls.enable_graph_filters.observe(lambda _: self.refresh())
        self.filter_widget.filter_controls.apply_button.on_click(lambda _: self.apply_filters())

        super().__init__(children=children)

    def redraw_display(self):
        self.general_controls.render_button.disabled = True
        self.general_controls.render_button.description = "Rendering..."
        self.general_controls.render_button.button_style = "info"
        self.general_controls.render_button.icon = "gear spin lg"
        self.general_controls.error_message.layout.visibility = "hidden"

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
        except Exception as e:
            self.general_controls.render_button.button_style = "danger"
            self.general_controls.render_button.icon = "exclamation-triangle"
            error_trace = traceback.format_exc().replace('\n', '<br>')
            self.general_controls.error_message.value = f"Error: {error_trace}"
            self.general_controls.error_message.layout.visibility = "visible"

        self.general_controls.render_button.disabled = False
        self.general_controls.render_button.description = "Render"

    def refresh(self):
        if self.general_controls.enable_graph_filters.value:
            self.children = [
                self.feature_controls,
                self.layer_controls,
                self.general_controls,
                self.filter_widget,
                self.display,
            ]
        else:
            self.children = [
                self.feature_controls,
                self.layer_controls,
                self.general_controls,
                self.display,
            ]

    def apply_filters(self):
        filtered_features = self.filter_widget.get_filtered_features()
        self.feature_controls.features = filtered_features.tolist() \
            if filtered_features is not None else list(range(self.num_features))

        self.feature_controls.input_radio.value = "Input"
        self.feature_controls.input_radio_changed({'new': 'Input'})  # In case it was already "Input"
