import traceback
from typing import Callable, List, Union

import ipywidgets as widgets
import torch

from .display_utils import display_features
from .filter_widget import FilterWidget
from .control_widgets import FeatureControls, LayerControls, GeneralControls


class InspectorWidget(widgets.VBox):
    def __init__(
            self,
            inspector,
    ):
        self.num_features = inspector.num_features
        self.num_layers = inspector.num_layers
        self.inspector = inspector

        self.feature_controls = FeatureControls(self.num_features)
        self.layer_controls = LayerControls(self.num_layers)
        self.general_controls = GeneralControls(inspector.bookmarked_features, self.apply_bookmarked)
        self.filter_widget = FilterWidget(
            inspector.feature_occurrences,
            torch.stack([feature.examples.activations.mean() for feature in inspector.feature_data]).detach().to(
                dtype=torch.float32),
            inspector.possible_occurrences
        )
        self.display = self.display_features(
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
            self.display = self.display_features(
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

        features = [self.inspector.feature_data[feature] for feature in features]

        return display_features(features, layers, self.inspector.sequences, self.bookmark_feature,
                                examples_per_layer)

    def bookmark_feature(self, feature: int):
        self.inspector.bookmarked_features.add(feature)
        self.general_controls.bookmark_viewer.update_bookmarks(list(self.inspector.bookmarked_features))

    def apply_filters(self):
        filtered_features = self.filter_widget.get_filtered_features()
        self.feature_controls.features = filtered_features.tolist() \
            if filtered_features is not None else list(range(self.num_features))

        self.feature_controls.input_radio.value = "Input"
        self.feature_controls.input_radio_changed({'new': 'Input'})  # In case it was already "Input"

    def apply_bookmarked(self, selected_bookmarks: List[int]):
        self.feature_controls.features = list(selected_bookmarks)
        self.feature_controls.input_radio.value = "Input"
        self.feature_controls.input_radio_changed({'new': 'Input'})
