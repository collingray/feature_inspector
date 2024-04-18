from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import torch
from IPython.display import display

SLIDER_WIDTH = "653px"
SLIDER_MARGINS = "0px 0px 0px 72px"


class FeatureFrequencyRange(widgets.VBox):
    def __init__(
            self,
            feature_occurrences: torch.Tensor,
            possible_occurrences: int,
            bins=100
    ):
        """
        :param feature_occurrences: A tensor of shape [num_layers, num_features], representing the number of times each
        feature is activated on each layer.
        :param possible_occurrences: The maximum number of times a feature could be activated.
        :param bins: The number of bins to use in the histogram.
        """
        self.num_activations = feature_occurrences.sum(dim=0)
        self.num_features = self.num_activations.shape[0]
        self.bins = bins

        self.log_freqs = (self.num_activations / possible_occurrences).log10().nan_to_num(neginf=-10)
        self.num_selected = torch.count_nonzero(self.num_activations).item()
        self.num_dead = self.num_features - self.num_selected

        self.min_freq = min([freq for freq in self.log_freqs if freq != -10]).item()
        self.max_freq = max(self.log_freqs).item()
        self.range = (self.min_freq, self.max_freq)

        # Indices of features which are in the current range
        self.selected_features = torch.where((self.min_freq <= self.log_freqs) & (self.log_freqs <= self.max_freq))[0]

        # Mask of features that are not filtered
        self.feature_mask = torch.ones(self.num_features, dtype=torch.bool)

        self.output = widgets.Output()

        self.slider = widgets.FloatRangeSlider(
            value=[self.min_freq, self.max_freq],
            min=self.min_freq,
            max=self.max_freq,
            step=(self.max_freq - self.min_freq) / bins,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
        )

        self.slider.layout.width = SLIDER_WIDTH
        self.slider.layout.margin = SLIDER_MARGINS

        self.feature_info = widgets.HBox([])

        self.update_plot({'new': self.slider.value})
        self.slider.observe(self.update_plot, 'value')

        layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            width='100%'
        )

        super().__init__(children=[self.output, self.slider, self.feature_info], layout=layout)

    def update_feature_info(self):
        left, right = self.slider.value
        self.num_selected = torch.count_nonzero((left <= self.log_freqs) & (self.log_freqs <= right)).item()
        self.feature_info.children = [
            widgets.Label(f"Total: {self.num_features}"),
            widgets.Label(f"Dead: {self.num_dead}"),
            widgets.Label(f"Selected: {self.num_selected}")
        ]

    def update_plot(self, change):
        self.update_feature_info()

        with (self.output):
            left, right = change['new']

            self.selected_features = torch.where(
                (left <= self.log_freqs) & (self.log_freqs <= right)
            )[0]

            fig = plt.figure(figsize=(8, 2))
            _, edges, p1 = plt.hist(self.log_freqs, bins=self.bins, range=self.range, color="lightgray")
            _, _, p2 = plt.hist(self.log_freqs * self.feature_mask, bins=self.bins, range=self.range, color="gray")

            for i in range(len(p1)):
                if (left <= edges[i]) & (edges[i + 1] <= right):
                    p1[i].set_facecolor('lightblue')
                    p2[i].set_facecolor('blue')

            fig.axes[0].tick_params(labelleft=True, labelright=True)
            plt.title('Log frequency density of features')

            self.output.clear_output(wait=True)
            plt.show()

    def refresh(self):
        self.update_plot({'new': self.slider.value})

    # Update the feature mask to include all features that are in 'filtered_features'. If 'filtered_features' is None,
    # all features are included.
    def update_filtered_features(self, filtered_features: Optional[torch.Tensor]):
        self.feature_mask[:] = False
        self.feature_mask[filtered_features] = True

        self.update_plot({'new': self.slider.value})


class LayersActivatedRange(widgets.VBox):
    def __init__(
            self,
            feature_occurrences: torch.Tensor,
    ):
        """
        :param feature_occurrences: A tensor of shape [num_layers, num_features], representing the number of times each
        feature is activated on each layer.
        """
        self.num_layers = feature_occurrences.shape[0]
        self.num_features = feature_occurrences.shape[1]
        self.bins = list(range(1, self.num_layers + 2))

        # The number of layers that each feature is activated on
        self.num_layers_activated = feature_occurrences.count_nonzero(dim=0)

        self.bincount = self.num_layers_activated.bincount()
        self.num_dead = self.bincount[0].item()
        self.bincount = self.bincount[1:]
        self.num_selected = self.bincount.sum().item()

        # Indices of features which are in the current range
        self.selected_features = torch.where(self.num_layers_activated > 0)[0]

        # Mask of features that are not filtered
        self.feature_mask = torch.ones(self.num_features, dtype=torch.bool)

        self.output = widgets.Output()

        self.slider = widgets.IntRangeSlider(
            value=[0, self.num_layers + 1],
            min=1,
            max=self.num_layers + 1,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )

        self.slider.layout.width = SLIDER_WIDTH
        self.slider.layout.margin = SLIDER_MARGINS

        self.feature_info = widgets.HBox([])

        self.update_plot({'new': self.slider.value})
        self.slider.observe(self.update_plot, 'value')

        layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            width='100%'
        )

        super().__init__(children=[self.output, self.slider, self.feature_info], layout=layout)

    def update_feature_info(self):
        left, right = self.slider.value
        self.num_selected = self.bincount[left - 1:right - 1].sum().item()
        self.feature_info.children = [
            widgets.Label(f"Total: {self.num_features}"),
            widgets.Label(f"Dead: {self.num_dead}"),
            widgets.Label(f"Selected: {self.num_selected}")
        ]

    def update_plot(self, change):
        self.update_feature_info()

        with (self.output):
            left, right = change['new']

            self.selected_features = torch.where(
                (left <= self.num_layers_activated) & (self.num_layers_activated < right)
            )[0]

            fig = plt.figure(figsize=(8, 2))
            _, _, patches1 = plt.hist(self.num_layers_activated, bins=self.bins, color="lightgray")
            _, _, patches2 = plt.hist(self.num_layers_activated * self.feature_mask, bins=self.bins, color="gray")

            for i in range(len(patches1)):
                if left - 1 <= i < right - 1:
                    patches1[i].set_facecolor('lightblue')
                    patches2[i].set_facecolor('blue')

            fig.axes[0].tick_params(labelleft=True, labelright=True)
            plt.title(f'Number of layers activated on by each feature')

            self.output.clear_output(wait=True)
            plt.show()

    def refresh(self):
        self.update_plot({'new': self.slider.value})

    # Update the feature mask to include all features that are in 'filtered_features'. If 'filtered_features' is None,
    # all features are included.
    def update_filtered_features(self, filtered_features: Optional[torch.Tensor]):
        self.feature_mask[:] = False
        self.feature_mask[filtered_features] = True

        self.update_plot({'new': self.slider.value})
