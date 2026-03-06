import pygame
import pygame_gui
from pygame_gui.core import UIElement


class ControlPanel(UIElement):
    """Control panel containing organized sections"""

    def __init__(self, relative_rect, manager):
        super().__init__(
            relative_rect=relative_rect,
            manager=manager,
            container=None,
            starting_height=1,
            layer_thickness=1,
        )

        panel_x = relative_rect.x
        panel_y = relative_rect.y
        panel_width = relative_rect.width

        # Create sub-panels for different sections
        self.joint_controls_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(panel_x, panel_y, panel_width, 200),
            starting_height=2,
            manager=manager,
        )

        self.actions_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(panel_x, panel_y + 210, panel_width, 150),
            starting_height=2,
            manager=manager,
        )

        self.info_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(panel_x, panel_y + 370, panel_width, 120),
            starting_height=2,
            manager=manager,
        )

        # Add widgets to each panel (example)
        self.joint1_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(10, 50, panel_width - 20, 20),
            start_value=0,
            value_range=(-180, 180),
            manager=manager,
            container=self.joint_controls_panel,
        )

        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, 10, panel_width - 20, 40),
            text="Reset",
            manager=manager,
            container=self.actions_panel,
        )

        self.info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 10, panel_width - 20, 30),
            text="Info goes here",
            manager=manager,
            container=self.info_panel,
        )
