import pygame_gui
import pygame
from pygame_gui.core import UIElement


class Canvas(UIElement):
    def __init__(self, relative_rect, manager, on_draw=None, on_click=None):

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
        self.on_click = on_click

    def draw_content(self):
        self.image.fill((255, 255, 255))

        if self.on_draw:
            self.on_draw(self.image)

    def process_event(self, event):
        """Called by manager when events happen"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                # Convert to canvas-local coordinates
                local_x = event.pos[0] - self.rect.x
                local_y = event.pos[1] - self.rect.y

                if self.on_click:
                    self.on_click(local_x, local_y, event.button)
                return True
        return False
