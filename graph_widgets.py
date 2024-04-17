from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import torch
from IPython.display import display


class HistogramRange(widgets.VBox):
    def __init__(
            self,
            frequency_occurrences: torch.Tensor,
            possible_occurrences: int,
            filtered_features: Optional[torch.Tensor] = None,
            bins=100
    ):
        """
        :param frequency_occurrences: A tensor of shape [num_layers, num_features], representing the number of times each
        feature is activated on each layer.
        :param possible_occurrences: The maximum number of times a feature could be activated.
        :param filtered_features: A tensor of feature indices to filter, or None to show all features.
        :param bins: The number of bins to use in the histogram.
        """
        self.counts = frequency_occurrences.sum(dim=0)
        self.log_freqs = [np.log10(count / possible_occurrences) for count in self.counts if count > 0]
        self.num_dead = len(self.counts) - len(self.log_freqs)
        self.num_selected = len(self.log_freqs)

        self.min_freq = min(self.log_freqs)
        self.max_freq = max(self.log_freqs)

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

        self.slider.layout.width = f"660px"
        self.slider.layout.margin = f"0px 0px 0px 120px"

        self.bins = np.linspace(self.min_freq, self.max_freq, bins)
        self.bin_width = self.bins[1] - self.bins[0]

        self.update_plot({'new': self.slider.value})

        self.slider.observe(self.update_plot, 'value')

        self.feature_info = widgets.HBox([
            widgets.Label(f"Total: {len(self.counts)}"),
            widgets.Label(f"Dead: {self.num_dead}"),
            widgets.Label(f"Selected: {self.num_selected}")
        ])

        layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            width='100%'
        )

        super().__init__(children=[self.output, self.slider, self.feature_info], layout=layout)

    def update_plot(self, change):
        with (self.output):
            self.output.clear_output(wait=True)
            # self.ax.clear()
            selected_range = change['new']
            left = selected_range[0]
            right = selected_range[1]
            plt.figure(figsize=(8, 2))
            _, edges, patches = plt.hist(self.log_freqs, bins=self.bins, color="gray")
            for i in range(len(patches)):
                if left <= (edges[i] + edges[i + 1]) / 2 <= right:
                    patches[i].set_facecolor('blue')
            plt.title(f'Log frequencies of Features')
            plt.show()

            self.num_selected = len([f for f in self.log_freqs if left <= f <= right])
            self.feature_info.children = [
                widgets.Label(f"Total: {len(self.log_freqs) + self.num_dead}"),
                widgets.Label(f"Dead: {self.num_dead}"),
                widgets.Label(f"Selected: {self.num_selected}")
            ]


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

        # Features which are not selected by the current range, ignoring the filtered features
        self.unselected_features = torch.tensor([])

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

        self.slider.layout.width = f"653px"
        self.slider.layout.margin = f"0px 0px 0px 80px"

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
            self.output.clear_output(wait=True)

            left, right = change['new']

            self.unselected_features[:] = torch.where(
                (self.num_layers_activated < left) & (self.num_layers_activated >= right)
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
            plt.show()

    def update_filtered_features(self, filtered_features: Optional[torch.Tensor]):
        self.feature_mask[:] = True
        if filtered_features is not None:
            self.feature_mask[filtered_features] = False

        self.update_plot({'new': self.slider.value})
