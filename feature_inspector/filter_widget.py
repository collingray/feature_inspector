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

        self.graph_filters = (
            FeatureFrequencyGraph(feature_occurrences, possible_occurrences),
            AverageActivationGraph(average_activations),
            LayersActivatedGraph(feature_occurrences)
        )

        self.filter_toggles = (
            self.filter_controls.enable_frequency_filter,
            self.filter_controls.enable_activation_filter,
            self.filter_controls.enable_layers_filter
        )

        for i in range(len(self.graph_filters)):
            self.graph_filters[i].layout.padding = "10px"
            self.graph_filters[i].slider.observe(lambda _, i=i: self.set_graph_filter(i), 'value')
            self.filter_toggles[i].observe(lambda _, i=i: (self.redraw_graphs(), self.set_graph_filter(i)), 'value')

        super().__init__(children=[])

        self.redraw_graphs()

    def get_filtered_features(self, exclude: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get the indices of features that are currently enabled by the filters.
        :param exclude: An index of a filter to exclude from the calculation.
        :return: A tensor of indices of features that are currently enabled, or None if all features are enabled.
        """
        filters = [graph.selected_features for i, graph in enumerate(self.graph_filters) if
                   self.filter_toggles[i].value and i != exclude]

        if len(filters) == 0:
            return None
        else:
            return torch.tensor(reduce(np.intersect1d, filters))

    def redraw_graphs(self):
        self.children = [self.filter_controls] + [graph for i, graph in enumerate(self.graph_filters) if
                                                  self.filter_toggles[i].value]

        for graph in self.graph_filters:
            graph.refresh()

    def set_graph_filter(self, i):
        for j in range(len(self.graph_filters)):
            if j != i:
                self.graph_filters[j].update_filtered_features(self.get_filtered_features(exclude=j))

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
