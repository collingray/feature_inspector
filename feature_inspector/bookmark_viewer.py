import ipywidgets as widgets
from typing import List, Callable


class BookmarkViewer(widgets.VBox):
    def __init__(self, bookmarks, on_click: Callable[[List[int]], None]):
        self.bookmarks = bookmarks
        self.on_click = on_click

        self.select_multiple = widgets.SelectMultiple(
            options=bookmarks,
            rows=5,
            layout=widgets.Layout(width='150px')
        )

        self.button = widgets.Button(description="Apply features", button_style='info')
        self.button.layout = widgets.Layout(width='150px')
        self.button.on_click(lambda _: self.on_click(self.select_multiple.value))
        self.button.disabled = True

        self.select_multiple.observe(self.on_selection_change, names='value')

        super().__init__(children=[self.select_multiple, self.button])

    def on_selection_change(self, change):
        if change['new']:
            self.button.disabled = False
            self.button.description = f"Apply {len(change['new'])} features"
        else:
            self.button.disabled = True
            self.button.description = "Apply features"

    def update_bookmarks(self, bookmarks):
        self.bookmarks = bookmarks
        self.select_multiple.options = bookmarks
