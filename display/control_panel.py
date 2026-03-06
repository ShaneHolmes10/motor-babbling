import pygame
import pygame_gui
from pygame_gui.core import UIElement


class ControlPanel(UIElement):
    """Control panel that manages widgets and their callbacks"""

    def __init__(self, relative_rect, manager, widgets, callbacks):
        super().__init__(
            relative_rect=relative_rect,
            manager=manager,
            container=None,
            starting_height=1,
            layer_thickness=1,
        )

        self.widgets = widgets
        self.callbacks = callbacks

    def process_event(self, event):
        """Handle widget events and fire callbacks"""
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            for widget_id, widget in self.widgets.items():
                if event.ui_element == widget:
                    callback = self.callbacks.get(f"{widget_id}_change")
                    if callback:
                        callback(event.ui_element.get_current_value())

        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            for widget_id, widget in self.widgets.items():
                if event.ui_element == widget:
                    callback = self.callbacks.get(f"{widget_id}_click")
                    if callback:
                        callback()

        return False

    def get_widget(self, widget_id):
        return self.widgets.get(widget_id)

    def set_slider_value(self, slider_id, value):
        slider = self.widgets.get(slider_id)
        if slider:
            slider.set_current_value(value)


class ControlPanelBuilder:
    """Builder for creating control panels with pygame_gui elements"""

    def __init__(self, x, y, width, height, manager):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.manager = manager

        self.current_y = y
        self.padding = 20
        self.spacing = 10

        self.widgets = {}
        self.callbacks = {}
        self.current_section = None

    def add_section(self, title, height):
        """Add a titled section panel"""
        section = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(self.x, self.current_y, self.width, height),
            starting_height=2,
            manager=self.manager,
        )

        if title:
            title_label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(10, 5, self.width - 20, 30),
                text=title,
                manager=self.manager,
                container=section,
            )

        self.current_section = {
            "panel": section,
            "current_y": 40 if title else 10,
            "start_y": self.current_y,
        }

        self.current_y += height + self.spacing
        return self

    def add_slider(
        self, slider_id, min_val, max_val, label=None, initial=0, on_change=None
    ):
        """Add a slider to the current section"""
        if not self.current_section:
            raise ValueError("Must call add_section first")

        section = self.current_section["panel"]
        y = self.current_section["current_y"]

        if label:
            label_widget = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(10, y, self.width - 40, 25),
                text=label,
                manager=self.manager,
                container=section,
            )
            y += 30

        slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, y, self.width - 40, 20),
            start_value=initial,
            value_range=(min_val, max_val),
            manager=self.manager,
            container=section,
        )

        self.widgets[slider_id] = slider
        if on_change:
            self.callbacks[f"{slider_id}_change"] = on_change

        self.current_section["current_y"] = y + 30
        return self

    def add_button(self, button_id, text, on_click=None, color=None):
        """Add a button to the current section"""
        if not self.current_section:
            raise ValueError("Must call add_section first")

        section = self.current_section["panel"]
        y = self.current_section["current_y"]

        button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, y, self.width - 40, 40),
            text=text,
            manager=self.manager,
            container=section,
        )

        self.widgets[button_id] = button
        if on_click:
            self.callbacks[f"{button_id}_click"] = on_click

        self.current_section["current_y"] = y + 45
        return self

    def add_label(self, label_id, text):
        """Add a label to the current section"""
        if not self.current_section:
            raise ValueError("Must call add_section first")

        section = self.current_section["panel"]
        y = self.current_section["current_y"]

        label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, y, self.width - 40, 25),
            text=text,
            manager=self.manager,
            container=section,
        )

        self.widgets[label_id] = label
        self.current_section["current_y"] = y + 30
        return self

    def add_text_box(self, textbox_id, html_text, height=100):
        """Add a text box for displaying info"""
        if not self.current_section:
            raise ValueError("Must call add_section first")

        section = self.current_section["panel"]
        y = self.current_section["current_y"]

        textbox = pygame_gui.elements.UITextBox(
            html_text=html_text,
            relative_rect=pygame.Rect(10, y, self.width - 40, height),
            manager=self.manager,
            container=section,
        )

        self.widgets[textbox_id] = textbox
        self.current_section["current_y"] = y + height + 10
        return self

    def build(self):
        """Return the constructed control panel"""
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        panel = ControlPanel(panel_rect, self.manager, self.widgets, self.callbacks)
        return panel
