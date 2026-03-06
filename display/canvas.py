import pygame_gui
import pygame
from pygame_gui.core import UIElement


class Canvas(UIElement):
    def __init__(self, relative_rect, manager, on_draw=None):

        # fmt: off
        super().__init__(
            relative_rect=relative_rect,
            manager=manager,
            container=None,
            starting_height=1,
            layer_thickness=1
        )
        # fmt: on

        self.image = pygame.Surface(relative_rect.size)
        self.rect = relative_rect
        self.on_draw = on_draw

    def draw_content(self):
        self.image.fill((255, 255, 255))

        if self.on_draw:
            self.on_draw(self.image)
