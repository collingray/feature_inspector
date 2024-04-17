import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display


class HistogramRange(widgets.VBox):
    def __init__(self, counts, total, bins=100):
        self.counts = counts
        self.log_freqs = [np.log10(count / total) for count in counts if count > 0]
        self.num_dead = len(counts) - len(self.log_freqs)
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
                if left <= (edges[i]+edges[i+1])/2 <= right:
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
    def __init__(self, feature_occurrences):
        self.num_layers = feature_occurrences.shape[0]
        self.num_features = feature_occurrences.shape[1]

        self.num_layers_activated = feature_occurrences.count_nonzero(dim=0)
        self.bincount = self.num_layers_activated.bincount()
        self.num_dead = self.bincount[0].item()
        self.bincount = self.bincount[1:]
        self.num_selected = self.bincount.sum().item()

        self.output = widgets.Output()

        self.slider = widgets.IntRangeSlider(
            value=[0, self.num_layers],
            min=1,
            max=self.num_layers+1,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )

        self.slider.layout.width = f"653px"
        self.slider.layout.margin = f"0px 0px 0px 80px"

        self.update_plot({'new': self.slider.value})

        self.slider.observe(self.update_plot, 'value')

        self.feature_info = widgets.HBox([
            widgets.Label(f"Total: {self.num_dead + self.num_selected}"),
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

            left, right = change['new']

            fig = plt.figure(figsize=(8, 2))
            _, edges, patches = plt.hist(self.num_layers_activated, bins=list(range(1, self.num_layers+2)), color="gray")
            for i in range(len(patches)):
                if left-1 <= i < right-1:
                    patches[i].set_facecolor('blue')
            fig.axes[0].tick_params(labelleft=True, labelright=True)
            plt.title(f'Number of layers activated on by each feature')
            plt.show()

            self.num_selected = self.bincount[left-1:right-1].sum().item()
            self.feature_info.children = [
                widgets.Label(f"Total: {self.num_features}"),
                widgets.Label(f"Dead: {self.num_dead}"),
                widgets.Label(f"Selected: {self.num_selected}")
            ]
