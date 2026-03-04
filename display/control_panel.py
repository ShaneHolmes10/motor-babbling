import pygame
from .ui_components import Slider, Button, Label, Container, Component


class ControlPanel(Component):
    background_color = (42, 42, 42)

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "control_panel")
        self.components = {}
        self.component_list = []

    def add_component(self, component_id, component):
        self.components[component_id] = component
        self.component_list.append(component)

    def draw(self, screen):
        pygame.draw.rect(
            screen, self.background_color, (self.x, self.y, self.width, self.height)
        )

        for component in self.component_list:
            component.draw(screen)

    def handle_event(self, event):
        for component in self.component_list:
            component.handle_event(event)

    def get_component(self, component_id):
        return self.components.get(component_id)

    def get_slider_value(self, slider_id):
        for comp in self.component_list:
            if isinstance(comp, Container):
                for sub_comp in comp.components:
                    if isinstance(sub_comp, Slider) and sub_comp.id == slider_id:
                        return sub_comp.value
            elif isinstance(comp, Slider) and comp.id == slider_id:
                return comp.value
        return None

    def set_slider_value(self, slider_id, value):
        for comp in self.component_list:
            if isinstance(comp, Container):
                for sub_comp in comp.components:
                    if isinstance(sub_comp, Slider) and sub_comp.id == slider_id:
                        sub_comp.set_value(value)
                        return
            elif isinstance(comp, Slider) and comp.id == slider_id:
                comp.set_value(value)
                return

    def check_button(self, button_id, event):
        for comp in self.component_list:
            if isinstance(comp, Container):
                for sub_comp in comp.components:
                    if isinstance(sub_comp, Button) and sub_comp.id == button_id:
                        return sub_comp.handle_event(event)
            elif isinstance(comp, Button) and comp.id == button_id:
                return comp.handle_event(event)
        return False


class ControlPanelBuilder:
    component_spacing = 10
    panel_padding = 20

    def __init__(self, x, y, width, height):
        self.panel = ControlPanel(x, y, width, height)
        self.current_y = y
        self.container_stack = []
        self.base_x = x
        self.base_width = width

    def begin_container(self, container_id, height, title=None):
        container_x = self.base_x + self.panel_padding
        container_width = self.base_width - 2 * self.panel_padding

        container = Container(
            container_x, self.current_y, container_width, height, container_id, title
        )

        self.container_stack.append(container)
        self.current_y += height + self.component_spacing

        return self

    def end_container(self):
        if not self.container_stack:
            raise ValueError("No container to end")

        container = self.container_stack.pop()
        self.panel.add_component(container.id, container)

        return self

    def add_slider(
        self,
        slider_id,
        min_val,
        max_val,
        initial=0,
        label="Slider",
        width=None,
        on_value_change=None,  # CALLBACK ADDED
    ):
        if width is None:
            if self.container_stack:
                width = self.container_stack[-1].width - 2 * Container.content_padding_x
            else:
                width = self.base_width - 2 * self.panel_padding

        slider_x = self.base_x + self.panel_padding + Container.content_padding_x
        slider = Slider(
            slider_x,
            0,
            width,
            min_val,
            max_val,
            initial,
            label,
            slider_id,
            on_value_change,
        )  # CALLBACK ADDED

        if self.container_stack:
            self.container_stack[-1].add_component(slider)
        else:
            slider.y = self.current_y
            self.panel.add_component(slider_id, slider)
            self.current_y += 60

        return self

    def add_button(
        self,
        button_id,
        text,
        width=None,
        height=40,
        color=(74, 158, 255),
        on_click=None,
    ):  # CALLBACK ADDED
        if width is None:
            if self.container_stack:
                width = self.container_stack[-1].width - 2 * Container.content_padding_x
            else:
                width = self.base_width - 2 * self.panel_padding

        button_x = self.base_x + self.panel_padding + Container.content_padding_x
        button = Button(
            button_x, 0, width, height, text, button_id, color, on_click
        )  # CALLBACK ADDED

        if self.container_stack:
            self.container_stack[-1].add_component(button)
        else:
            button.y = self.current_y
            self.panel.add_component(button_id, button)
            self.current_y += height + self.component_spacing

        return self

    def add_label(self, label_id, text, font_size="small", color=(255, 255, 255)):
        label_x = self.base_x + self.panel_padding + Container.content_padding_x
        label = Label(label_x, 0, text, label_id, font_size, color)

        font_heights = {"small": 24, "medium": 28, "large": 36}
        label_height = font_heights.get(font_size, 24)

        if self.container_stack:
            self.container_stack[-1].add_component(label)
        else:
            label.y = self.current_y
            self.panel.add_component(label_id, label)
            self.current_y += label_height + self.component_spacing

        return self

    def add_spacing(self, amount):
        if self.container_stack:
            self.container_stack[-1].current_content_y += amount
        else:
            self.current_y += amount
        return self

    def build(self):
        if self.container_stack:
            raise ValueError("Unclosed containers remain")
        return self.panel
