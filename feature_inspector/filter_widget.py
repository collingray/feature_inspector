from functools import reduce
from typing import Optional

import ipywidgets as widgets
import torch
import numpy as np

from .graph_widgets import FeatureFrequencyGraph, LayersActivatedGraph, AverageActivationGraph


class FilterWidget(widgets.VBox):
    def __init__(
            self,
            feature_occurrences: torch.Tensor,
            average_activations: torch.Tensor,
            possible_occurrences: int,
    ):
        self.filter_controls = FilterControls()
        self.frequency_graph = FeatureFrequencyGraph(feature_occurrences, possible_occurrences)
        self.activation_graph = AverageActivationGraph(average_activations)
        self.layers_graph = LayersActivatedGraph(feature_occurrences)

        self.frequency_graph.layout.padding = "10px"
        self.activation_graph.layout.padding = "10px"
        self.layers_graph.layout.padding = "10px"

        self.filter_controls.enable_frequency_filter.observe(
            lambda _: (self.redraw_graphs(), self.set_frequency_filter()),
            'value'
        )
        self.filter_controls.enable_activation_filter.observe(
            lambda _: (self.redraw_graphs(), self.set_activation_filter()),
            'value'
        )
        self.filter_controls.enable_layers_filter.observe(
            lambda _: (self.redraw_graphs(), self.set_layers_filter()),
            'value'
        )

        self.frequency_graph.slider.observe(lambda _: self.set_frequency_filter(), 'value')
        self.activation_graph.slider.observe(lambda _: self.set_activation_filter(), 'value')
        self.layers_graph.slider.observe(lambda _: self.set_layers_filter(), 'value')

        super().__init__(children=[])

        self.redraw_graphs()

    def get_filtered_features(self) -> Optional[torch.Tensor]:
        """
        :return: A tensor of indices of features that are currently enabled, or None if all features are enabled.
        """
        freq_filter = self.frequency_graph.selected_features if self.filter_controls.enable_frequency_filter else None
        activation_filter = self.activation_graph.selected_features if self.filter_controls.enable_activation_filter else None
        layer_filter = self.layers_graph.selected_features if self.filter_controls.enable_layers_filter else None

        if freq_filter is None and activation_filter is None and layer_filter is None:
            return None
        else:
            return torch.tensor(reduce(np.intersect1d, [x for x in [freq_filter, activation_filter, layer_filter] if
                                                        x is not None]))

        # if (freq_filter is not None) and (layer_filter is not None):
        #     return torch.tensor(np.intersect1d(freq_filter, layer_filter))
        # elif freq_filter is None:
        #     return layer_filter
        # else:
        #     return freq_filter

    def redraw_graphs(self):
        self.children = [
            graph for graph in [
                self.filter_controls,
                self.frequency_graph if self.filter_controls.enable_frequency_filter.value else None,
                self.activation_graph if self.filter_controls.enable_activation_filter.value else None,
                self.layers_graph if self.filter_controls.enable_layers_filter.value else None
            ] if graph is not None
        ]

        self.frequency_graph.refresh()
        self.activation_graph.refresh()
        self.layers_graph.refresh()

    def set_frequency_filter(self):
        enabled = self.filter_controls.enable_frequency_filter.value
        self.activation_graph.update_filtered_features(self.frequency_graph.selected_features if enabled else None)
        self.layers_graph.update_filtered_features(self.frequency_graph.selected_features if enabled else None)

        self.filter_controls.apply_button.description = f"Apply {self.get_filtered_features().size(0)} features"

    def set_activation_filter(self):
        enabled = self.filter_controls.enable_activation_filter.value
        self.frequency_graph.update_filtered_features(self.activation_graph.selected_features if enabled else None)
        self.layers_graph.update_filtered_features(self.activation_graph.selected_features if enabled else None)

        self.filter_controls.apply_button.description = f"Apply {self.get_filtered_features().size(0)} features"

    def set_layers_filter(self):
        enabled = self.filter_controls.enable_layers_filter.value
        self.frequency_graph.update_filtered_features(self.layers_graph.selected_features if enabled else None)
        self.activation_graph.update_filtered_features(self.layers_graph.selected_features if enabled else None)

        self.filter_controls.apply_button.description = f"Apply {self.get_filtered_features().size(0)} features"


class FilterControls(widgets.HBox):
    def __init__(self):
        self.enable_frequency_filter = widgets.Checkbox(
            value=False,
            description='Enable frequency filter',
            disabled=False,
        )

        self.enable_activation_filter = widgets.Checkbox(
            value=False,
            description='Enable activation filter',
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

        super().__init__(
            children=[self.enable_frequency_filter, self.enable_activation_filter, self.enable_layers_filter,
                      self.apply_button])
