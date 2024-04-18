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

        children = [
            self.feature_frequency,
            self.layers_activated
        ]

        self.filter_controls.enable_frequency_filter.observe(self.set_frequency_filter, 'value')
        self.filter_controls.enable_layers_filter.observe(self.set_layers_filter, 'value')

        super().__init__(children=children)

    def get_filtered_features(self) -> Optional[torch.Tensor]:
        """
        :return: A tensor of indices of features that are currently enabled, or None if all features are enabled.
        """
        freq_filter = self.feature_frequency.selected_features if self.filter_controls.enable_frequency_filter else None
        layer_filter = self.layers_activated.selected_features if self.filter_controls.enable_layers_filter else None

        return reduce(
            np.intersect1d,
            [filter for filter in [freq_filter, layer_filter] if filter is not None],
            None
        )

    def redraw_graphs(self):
        self.children = [
            graph for graph in [
                self.feature_frequency if self.filter_controls.enable_frequency_filter else None,
                self.layers_activated if self.filter_controls.enable_layers_filter else None
            ] if graph is not None
        ]

    def set_frequency_filter(self, change):
        enabled = change['new']
        self.layers_activated.update_filtered_features(self.feature_frequency.selected_features if enabled else None)
        self.redraw_graphs()

    def set_layers_filter(self, change):
        enabled = change['new']
        self.feature_frequency.update_filtered_features(self.layers_activated.selected_features if enabled else None)
        self.redraw_graphs()


class FilterControls(widgets.HBox):
    def __init__(self):
        self.enable_frequency_filter = widgets.Checkbox(
            value=True,
            description='Enable frequency filter',
            disabled=False,
        )

        self.enable_layers_filter = widgets.Checkbox(
            value=True,
            description='Enable layers filter',
            disabled=False,
        )

        self.apply_button = widgets.Button(
            description="Apply filtered features",
            button_style="info"
        )

        super().__init__(children=[self.enable_frequency_filter, self.enable_layers_filter, self.apply_button])
