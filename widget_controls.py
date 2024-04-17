from typing import List

import ipywidgets as widgets


class FeatureControls(widgets.VBox):
    def __init__(self, num_features: int):
        self.num_features: int = num_features
        self.features: List[int] = [0, 1, 2]

        self.input_radio = widgets.RadioButtons(
            options=['Range', 'Input'],
            disabled=False
        )

        self.slider = widgets.IntRangeSlider(
            value=[self.features[0], self.features[-1]],
            min=0,
            max=num_features - 1,
            step=1,
            disabled=False,
            continuous_update=False
        )

        self.text_input = widgets.Text(
            value=",".join(map(str, self.features)),
            disabled=True
        )

        self.input_radio.layout.width = "15%"
        self.slider.layout.width = "75%"
        self.text_input.layout.width = "auto"

        self.input_radio.observe(self.input_radio_changed, 'value')
        self.slider.observe(self.slider_changed, 'value')
        self.text_input.observe(self.text_input_changed, 'value')

        label = widgets.Label("Features")
        label.style.font_size = "16px"

        children = [
            label,
            widgets.HBox(children=[self.input_radio, self.slider, self.text_input])
        ]

        super().__init__(children=children)

    def input_radio_changed(self, change):
        if change['new'] == 'Range':
            self.slider.disabled = False
            self.text_input.disabled = True
            self.slider.value = (self.features[0], self.features[-1])
        else:
            self.slider.disabled = True
            self.text_input.disabled = False
            self.text_input.value = ",".join(map(str, self.features))

    def slider_changed(self, change):
        self.features = list(range(change['new'][0], change['new'][1] + 1))

    def text_input_changed(self, change):
        try:
            self.features = list(map(int, change['new'].split(",")))
            self.features.sort()
        except:
            pass


class LayerControls(widgets.VBox):
    def __init__(self, num_layers: int):
        self.num_layers: int = num_layers
        self.layers: List[int] = list(range(num_layers))

        self.input_radio = widgets.RadioButtons(
            options=['All', 'Range', 'Input'],
            disabled=False
        )

        self.slider = widgets.IntRangeSlider(
            value=[self.layers[0], self.layers[-1]],
            min=0,
            max=num_layers - 1,
            step=1,
            disabled=True,
            continuous_update=False
        )

        self.text_input = widgets.Text(
            value=",".join(map(str, self.layers)),
            disabled=True
        )

        self.input_radio.layout.width = "15%"
        self.slider.layout.width = "75%"
        self.text_input.layout.width = "auto"

        self.input_radio.observe(self.input_radio_changed, 'value')
        self.slider.observe(self.slider_changed, 'value')
        self.text_input.observe(self.text_input_changed, 'value')

        label = widgets.Label("Layers")
        label.style.font_size = "16px"

        children = [
            label,
            widgets.HBox(children=[self.input_radio, self.slider, self.text_input])
        ]

        super().__init__(children=children)

    def input_radio_changed(self, change):
        if change['new'] == 'All':
            self.slider.disabled = True
            self.text_input.disabled = True
            self.layers = list(range(self.num_layers))
        elif change['new'] == 'Range':
            self.slider.disabled = False
            self.text_input.disabled = True
            self.slider.value = (self.layers[0], self.layers[-1])
        else:
            self.slider.disabled = True
            self.text_input.disabled = False
            self.text_input.value = ",".join(map(str, self.layers))

    def slider_changed(self, change):
        self.layers = list(range(change['new'][0], change['new'][1] + 1))

    def text_input_changed(self, change):
        try:
            self.layers = list(map(int, change['new'].split(",")))
            self.layers.sort()
        except:
            pass


class GeneralControls(widgets.VBox):
    def __init__(self):
        self.examples_per_layer = widgets.IntSlider(
            value=3,
            min=1,
            max=10,
            step=1,
            description='Examples per layer:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style={"description_width": "initial", "width": "auto"}
        )

        self.enable_graph_filters = widgets.Checkbox(
            value=False,
            description='Enable graph filters',
            disabled=False,
            indent=False
        )

        self.render_button = widgets.Button(
            description='Render',
            disabled=False,
            button_style='info',
            tooltip='Render',
            icon='check'
        )

        label = widgets.Label("General")
        label.style.font_size = "16px"

        children = [
            label,
            self.examples_per_layer,
            self.enable_graph_filters,
            self.render_button
        ]

        super().__init__(children=children)
