from typing import Optional

import ipywidgets as widgets
import torch
import numpy as np
from functools import reduce

from graph_widgets import FeatureFrequencyRange, LayersActivatedRange


class FilterWidget(widgets.VBox):
    def __init__(
            self,
            feature_occurrences: torch.Tensor,
            possible_occurrences: int,
    ):
        self.filter_controls = FilterControls()
        self.feature_frequency = FeatureFrequencyRange(feature_occurrences, possible_occurrences)
        self.layers_activated = LayersActivatedRange(feature_occurrences)

        self.filter_controls.enable_frequency_filter.observe(lambda _: self.set_frequency_filter(), 'value')
        self.filter_controls.enable_layers_filter.observe(lambda _: self.set_layers_filter(), 'value')

        self.feature_frequency.slider.observe(lambda _: self.set_frequency_filter(), 'value')
        self.layers_activated.slider.observe(lambda _: self.set_layers_filter(), 'value')

        super().__init__(children=[])

        self.redraw_graphs()

    def get_filtered_features(self) -> Optional[torch.Tensor]:
        """
        :return: A tensor of indices of features that are currently enabled, or None if all features are enabled.
        """
        freq_filter = self.feature_frequency.selected_features if self.filter_controls.enable_frequency_filter else None
        layer_filter = self.layers_activated.selected_features if self.filter_controls.enable_layers_filter else None

        if (freq_filter is not None) and (layer_filter is not None):
            return torch.tensor(np.intersect1d(freq_filter, layer_filter))
        elif freq_filter is None:
            return layer_filter
        else:
            return freq_filter

    def redraw_graphs(self):
        self.children = [
            graph for graph in [
                self.filter_controls,
                self.feature_frequency if self.filter_controls.enable_frequency_filter.value else None,
                self.layers_activated if self.filter_controls.enable_layers_filter.value else None
            ] if graph is not None
        ]

    def set_frequency_filter(self):
        enabled = self.filter_controls.enable_frequency_filter.value
        self.layers_activated.update_filtered_features(self.feature_frequency.selected_features if enabled else None)

        self.redraw_graphs()
        self.feature_frequency.refresh()

    def set_layers_filter(self):
        enabled = self.filter_controls.enable_layers_filter.value
        self.feature_frequency.update_filtered_features(self.layers_activated.selected_features if enabled else None)

        self.redraw_graphs()
        self.layers_activated.refresh()


class FilterControls(widgets.HBox):
    def __init__(self):
        self.enable_frequency_filter = widgets.Checkbox(
            value=False,
            description='Enable frequency filter',
            disabled=False,
        )

        self.enable_layers_filter = widgets.Checkbox(
            value=False,
            description='Enable layers filter',
            disabled=False,
        )

        self.apply_button = widgets.Button(
            description="Apply filtered features",
            button_style="info",
        )

        super().__init__(children=[self.enable_frequency_filter, self.enable_layers_filter, self.apply_button])
