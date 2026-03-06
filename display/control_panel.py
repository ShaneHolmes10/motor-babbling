import pygame
import pygame_gui
from pygame_gui.core import UIElement


class ControlPanel(UIElement):
    """
    Control panel that manages widgets and their callbacks.
    Acts as a coordinator between pygame_gui widgets and application code.
    """

    def __init__(self, relative_rect, manager, widgets, callbacks):
        """
        Initialize the control panel.

        Args:
            relative_rect: pygame.Rect defining position and size
            manager: pygame_gui UIManager instance
            widgets: Dict mapping widget IDs to widget instances
            callbacks: Dict mapping callback keys to callback functions
        """

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
        """
        Handle pygame_gui widget events and fire appropriate callbacks.

        Args:
            event: pygame.event.Event to process

        Returns:
            False (events not consumed by panel itself)
        """

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
        """
        Retrieve a widget by its ID.

        Args:
            widget_id: String identifier for the widget

        Returns:
            Widget instance or None if not found
        """

        return self.widgets.get(widget_id)

    def set_slider_value(self, slider_id, value):
        """
        Set a slider's value programmatically.

        Args:
            slider_id: String identifier for the slider
            value: New value to set
        """

        slider = self.widgets.get(slider_id)
        if slider:
            slider.set_current_value(value)


class ControlPanelBuilder:
    """
    Builder for creating control panels with pygame_gui elements.
    Uses fluent interface for easy panel construction.
    """

    def __init__(self, x, y, width, height, manager):
        """
        Initialize the builder.

        Args:
            x: X position of the panel
            y: Y position of the panel
            width: Width of the panel
            height: Height of the panel
            manager: pygame_gui UIManager instance
        """

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
        """
        Add a scrollable section to the control panel.

        Args:
            title: Section title (None for no title)
            height: Height of the section in pixels

        Returns:
            self for method chaining
        """

        section = pygame_gui.elements.UIScrollingContainer(
            relative_rect=pygame.Rect(
                self.x, self.current_y, self.width, height
            ),
            manager=self.manager,
            starting_height=2,
        )

        if title:
            pygame_gui.elements.UILabel(
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

    def end_section(self):
        """
        Finalize the current section and set its scrollable area size.
        Must be called after adding widgets to a section.

        Returns:
            self for method chaining
        """

        if not self.current_section:
            raise ValueError("No section to end")

        section = self.current_section["panel"]
        content_height = self.current_section["current_y"] + 20

        # Tell the scrolling container how tall the content is

        section.set_scrollable_area_dimensions(
            (self.width - 20, content_height)
        )

        self.current_section = None
        return self

    def add_slider(
        self,
        slider_id,
        min_val,
        max_val,
        label=None,
        initial=0,
        on_change=None,
    ):
        """
        Add a horizontal slider to the current section.

        Args:
            slider_id: Unique identifier for this slider
            min_val: Minimum slider value
            max_val: Maximum slider value
            label: Optional label text displayed above slider
            initial: Initial slider value
            on_change: Optional callback function(value) called when slider
            moves

        Returns:
            self for method chaining
        """

        if not self.current_section:
            raise ValueError("Must call add_section first")

        section = self.current_section["panel"]
        y = self.current_section["current_y"]

        if label:
            pygame_gui.elements.UILabel(
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
        """
        Add a button to the current section.

        Args:
            button_id: Unique identifier for this button
            text: Button label text
            on_click: Optional callback function() called when button is
            clicked
            color: Optional color (currently unused)

        Returns:
            self for method chaining
        """

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
        """
        Add a text label to the current section.

        Args:
            label_id: Unique identifier for this label
            text: Label text content

        Returns:
            self for method chaining
        """

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
        """
        Add a text box for displaying formatted multi-line text.

        Args:
            textbox_id: Unique identifier for this text box
            html_text: HTML-formatted text content
            height: Height of the text box in pixels

        Returns:
            self for method chaining
        """

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
        """
        Construct and return the final ControlPanel instance.

        Returns:
            ControlPanel instance with all widgets and callbacks configured
        """

        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        panel = ControlPanel(
            panel_rect, self.manager, self.widgets, self.callbacks
        )
        return panel
