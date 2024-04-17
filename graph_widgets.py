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
            readout_format='d',
        )

        self.slider.layout.width = f"656px"
        self.slider.layout.margin = f"0px 0px 0px 105px"

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
