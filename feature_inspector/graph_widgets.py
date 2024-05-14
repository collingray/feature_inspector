from abc import ABC, abstractproperty, abstractmethod, ABCMeta
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import torch
from IPython.display import display, Image

SLIDER_WIDTH = "653px"
SLIDER_MARGINS = "0px 0px 0px 98px"


class GraphWidgetMeta(ABCMeta, type(widgets.VBox)):
    pass


class GraphWidget(ABC, widgets.VBox, metaclass=GraphWidgetMeta):
    def __init__(self, title: str, slider):
        self.output = widgets.Output()

        # Create the initial plot
        with self.output:
            self.fig, self.axes = plt.subplots(figsize=(8, 2))
            self.axes.tick_params(labelleft=True, labelright=True)
            self.fig.suptitle(title)
            plt.show()

        self.slider = slider
        self.slider.layout.width = SLIDER_WIDTH
        self.slider.layout.margin = SLIDER_MARGINS
        self.slider.observe(self.update_plot, 'value')

        self.feature_info = widgets.HBox([])

        layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            width='100%'
        )

        super().__init__(children=[self.output, self.slider, self.feature_info], layout=layout)

    def update_feature_info(self):
        self.feature_info.children = [
            widgets.Label(f"Total: {self.num_features}"),
            widgets.Label(f"Dead: {self.num_dead}"),
            widgets.Label(f"Selected: {len(self.selected_features)}")
        ]

    def refresh(self):
        self.update_plot({'new': self.slider.value})

    # Update the feature mask to include all features that are in 'filtered_features'. If 'filtered_features' is None,
    # all features are included.
    def update_filtered_features(self, filtered_features: Optional[torch.Tensor]):
        self.feature_mask[:] = False
        self.feature_mask[filtered_features] = True
        self.update_plot({'new': self.slider.value})

    @abstractmethod
    def update_plot(self, change):
        pass


class FeatureFrequencyRange(GraphWidget):
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

        self.log_freqs = (self.num_activations / possible_occurrences).log10().nan_to_num(neginf=-10)

        self.min_freq = min([freq for freq in self.log_freqs if freq != -10]).item()
        self.max_freq = max(self.log_freqs).item()
        self.bin_width = (self.max_freq - self.min_freq) / bins
        self.bins = list(np.arange(self.min_freq, self.max_freq + (2 * self.bin_width), self.bin_width))

        # Indices of features which are in the current range
        self.selected_features = torch.where((self.min_freq <= self.log_freqs) & (self.log_freqs <= self.max_freq))[0]

        # Number of features that are not activated at all
        self.num_dead = self.num_features - len(self.selected_features)

        # Mask of features that are not filtered
        self.feature_mask = torch.ones(self.num_features, dtype=torch.bool)

        slider = widgets.FloatRangeSlider(
            value=[self.min_freq, self.max_freq],
            min=self.min_freq,
            max=self.max_freq,
            step=self.bin_width,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
        )

        super().__init__('Log frequency density of features', slider)

    def update_plot(self, change):
        left, right = change['new']
        self.selected_features = torch.where(
            (left <= self.log_freqs) & (self.log_freqs <= right)
        )[0]

        self.update_feature_info()

        with self.output:
            self.axes.clear()
            _, edges, p1 = self.axes.hist(self.log_freqs, bins=self.bins, color="lightgray")
            _, _, p2 = self.axes.hist(self.log_freqs * self.feature_mask, bins=self.bins, color="gray")

            for i in range(len(p1)):
                center = (edges[i] + edges[i + 1]) / 2
                if (left < center) & (center < right):
                    p1[i].set_facecolor('lightblue')
                    p2[i].set_facecolor('blue')

            image_data = BytesIO()
            self.fig.savefig(image_data, format='png')
            image_data.seek(0)
            image = Image(image_data.read())
            self.output.clear_output(wait=True)
            display(image)


class LayersActivatedRange(GraphWidget):
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

        # Indices of features which are in the current range
        self.selected_features = torch.where(self.num_layers_activated > 0)[0]

        # Number of features that are not activated on any layer
        self.num_dead = self.num_features - len(self.selected_features)

        # Mask of features that are not filtered
        self.feature_mask = torch.ones(self.num_features, dtype=torch.bool)

        slider = widgets.IntRangeSlider(
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

        super().__init__('Number of layers activated by features', slider)

    def update_plot(self, change):
        left, right = change['new']
        self.selected_features = torch.where(
            (left <= self.num_layers_activated) & (self.num_layers_activated < right)
        )[0]

        self.update_feature_info()

        with self.output:
            self.axes.clear()
            _, _, p1 = self.axes.hist(self.num_layers_activated, bins=self.bins, color="lightgray")
            _, _, p2 = self.axes.hist(self.num_layers_activated * self.feature_mask, bins=self.bins, color="gray")

            for i in range(len(p1)):
                if left - 1 <= i < right - 1:
                    p1[i].set_facecolor('lightblue')
                    p2[i].set_facecolor('blue')

            image_data = BytesIO()
            self.fig.savefig(image_data, format='png')
            image_data.seek(0)
            image = Image(image_data.read())

            self.output.clear_output(wait=True)
            display(image)
