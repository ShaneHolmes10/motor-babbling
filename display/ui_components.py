import pygame
from abc import ABC, abstractmethod
import math


class Component(ABC):
    """Base class for all UI components"""

    def __init__(self, x, y, width, height, component_id):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.id = component_id
        self.label_offset = 0

    @abstractmethod
    def draw(self, screen):
        """Draw the component on the screen"""
        pass

    @abstractmethod
    def handle_event(self, event):
        """Handle pygame events"""
        pass


class Container(Component):
    background_color = (35, 35, 35)
    title_color = (74, 158, 255)
    title_size = 28
    title_padding = 10
    title_bottom_spacing = 20
    content_padding_x = 10
    content_padding_y = 10
    content_spacing = 10
    scrollbar_bg_color = (60, 60, 60)
    scrollbar_handle_color = (100, 100, 100)
    scrollbar_width = 10

    def __init__(self, x, y, width, height, component_id, title=None):
        super().__init__(x, y, width, height, component_id)
        self.title = title
        self.components = []
        self.scroll_offset = 0
        self.content_height = 0
        self.dragging_scrollbar = False
        self.scroll_handle_y = 0
        self.current_content_y = self._get_content_start_y()

    def _get_content_start_y(self):
        if self.title:
            return (
                self.y
                + self.title_padding
                + self.title_size
                + self.title_bottom_spacing
            )
        return self.y + self.content_padding_y

    def add_component(self, component):
        component.y = self.current_content_y + component.label_offset
        self.current_content_y += component.height + self.content_spacing

        self.components.append(component)
        self._recalculate_content_height()

    def _recalculate_content_height(self):
        if not self.components:
            self.content_height = 0
            return

        max_y = 0
        for comp in self.components:
            comp_bottom = comp.y + comp.height
            max_y = max(max_y, comp_bottom)

        self.content_height = max_y - self.y + self.content_padding_y

    def needs_scrollbar(self):
        return self.content_height > self.height

    def draw(self, screen):
        container_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.background_color, container_rect, border_radius=5)

        if self.title:
            title_surface = pygame.font.Font(None, self.title_size).render(
                self.title, True, self.title_color
            )
            screen.blit(
                title_surface,
                (self.x + self.title_padding, self.y + self.title_padding),
            )

        sub_surface = screen.subsurface(container_rect)

        for component in self.components:
            visible_y = component.y - self.y - self.scroll_offset

            if -50 < visible_y < self.height:
                temp_y = component.y
                component.y = visible_y
                component.draw(sub_surface)
                component.y = temp_y

        if self.needs_scrollbar():
            self._draw_scrollbar(screen)

    def _draw_scrollbar(self, screen):
        scrollbar_x = self.x + self.width - self.scrollbar_width
        scrollbar_rect = pygame.Rect(
            scrollbar_x, self.y, self.scrollbar_width, self.height
        )
        pygame.draw.rect(screen, self.scrollbar_bg_color, scrollbar_rect)

        visible_ratio = self.height / self.content_height
        handle_height = max(20, int(self.height * visible_ratio))

        max_scroll = self.content_height - self.height
        scroll_ratio = self.scroll_offset / max_scroll if max_scroll > 0 else 0

        handle_y_range = self.height - handle_height
        self.scroll_handle_y = self.y + int(scroll_ratio * handle_y_range)

        handle_rect = pygame.Rect(
            scrollbar_x, self.scroll_handle_y, self.scrollbar_width, handle_height
        )
        pygame.draw.rect(
            screen, self.scrollbar_handle_color, handle_rect, border_radius=3
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos

            if self.needs_scrollbar():
                scrollbar_x = self.x + self.width - self.scrollbar_width
                if scrollbar_x <= mouse_x <= self.x + self.width:
                    if self.y <= mouse_y <= self.y + self.height:
                        self.dragging_scrollbar = True
                        return

            if self.x <= mouse_x <= self.x + self.width - self.scrollbar_width:
                if self.y <= mouse_y <= self.y + self.height:
                    relative_x = mouse_x
                    relative_y = mouse_y + self.scroll_offset

                    for component in self.components:
                        temp_event = pygame.event.Event(
                            event.type,
                            pos=(relative_x, relative_y),
                            button=event.button if hasattr(event, "button") else None,
                        )
                        component.handle_event(temp_event)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_scrollbar = False

            mouse_x, mouse_y = event.pos
            if self.x <= mouse_x <= self.x + self.width - self.scrollbar_width:
                if self.y <= mouse_y <= self.y + self.height:
                    relative_x = mouse_x
                    relative_y = mouse_y - self.y + self.scroll_offset

                    for component in self.components:
                        temp_event = pygame.event.Event(
                            event.type,
                            pos=(relative_x, relative_y),
                            button=event.button if hasattr(event, "button") else None,
                        )
                        component.handle_event(temp_event)

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_scrollbar and self.needs_scrollbar():
                mouse_y = event.pos[1]

                visible_ratio = self.height / self.content_height
                handle_height = max(20, int(self.height * visible_ratio))
                handle_y_range = self.height - handle_height

                relative_y = mouse_y - self.y
                scroll_ratio = relative_y / handle_y_range if handle_y_range > 0 else 0
                scroll_ratio = max(0, min(1, scroll_ratio))

                max_scroll = self.content_height - self.height
                self.scroll_offset = int(scroll_ratio * max_scroll)
            else:
                mouse_x, mouse_y = event.pos
                if self.x <= mouse_x <= self.x + self.width - self.scrollbar_width:
                    if self.y <= mouse_y <= self.y + self.height:
                        relative_x = mouse_x
                        relative_y = mouse_y - self.y + self.scroll_offset

                        for component in self.components:
                            temp_event = pygame.event.Event(
                                event.type, pos=(relative_x, relative_y)
                            )
                            component.handle_event(temp_event)

        elif event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if self.x <= mouse_x <= self.x + self.width:
                if self.y <= mouse_y <= self.y + self.height:
                    if self.needs_scrollbar():
                        self.scroll_offset -= event.y * 20
                        max_scroll = self.content_height - self.height
                        self.scroll_offset = max(0, min(max_scroll, self.scroll_offset))


class Slider(Component):
    name_color = (255, 255, 255)
    name_size = 24
    value_color = (74, 158, 255)
    value_size = 24
    slider_color = (50, 50, 50)
    label_space = 25
    slider_height = 10
    handle_radius = 9

    def __init__(
        self,
        x,
        y,
        width,
        min_val,
        max_val,
        initial_val,
        label,
        component_id,
        on_value_change=None,  # CALLBACK ADDED
    ):
        total_height = self.label_space + self.slider_height + 25
        super().__init__(x, y, width, total_height, component_id)
        self.label_offset = self.label_space
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.on_value_change = on_value_change  # CALLBACK ADDED

    def draw(self, screen):
        text = pygame.font.Font(None, self.name_size).render(
            self.label, True, self.name_color
        )
        screen.blit(text, (self.x, self.y - self.label_space))

        value_text = pygame.font.Font(None, self.value_size).render(
            f"{int(self.value)}°", True, self.value_color
        )
        screen.blit(value_text, (self.x + self.width - 50, self.y - self.label_space))

        pygame.draw.rect(
            screen,
            self.slider_color,
            (self.x, self.y, self.width, self.slider_height),
            border_radius=5,
        )

        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
        usable_width = self.width - 2 * self.handle_radius
        handle_x = self.x - self.handle_radius + int(normalized * usable_width)
        pygame.draw.circle(
            screen,
            self.value_color,
            (handle_x, self.y + self.slider_height // 2),
            self.handle_radius,
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos

            # Check if click is anywhere on the slider track
            if (
                self.x <= mouse_x <= self.x + self.width
                and self.y - self.label_space
                <= mouse_y
                <= self.y + self.slider_height + 5
            ):

                # Snap to click position immediately
                normalized = (mouse_x - self.x) / self.width
                normalized = max(0, min(1, normalized))
                new_value = self.min_val + normalized * (self.max_val - self.min_val)

                if new_value != self.value:
                    self.value = new_value
                    if self.on_value_change:
                        self.on_value_change(self.value)

                # Start dragging
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = event.pos[0]
            normalized = (mouse_x - self.x) / self.width
            normalized = max(0, min(1, normalized))
            new_value = self.min_val + normalized * (self.max_val - self.min_val)

            if new_value != self.value:
                self.value = new_value
                if self.on_value_change:
                    self.on_value_change(self.value)

    def set_value(self, value):
        old_value = self.value
        self.value = max(self.min_val, min(self.max_val, value))
        # CALLBACK ADDED
        if self.value != old_value and self.on_value_change:
            self.on_value_change(self.value)


class Button(Component):
    name_color = (255, 255, 255)
    name_size = 24

    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        component_id,
        color=(74, 158, 255),
        on_click=None,  # CALLBACK ADDED
    ):
        self.rect = pygame.Rect(x, y, width, height)
        super().__init__(x, y, width, height, component_id)
        self.text = text
        self.color = color
        self.hover = False
        self.on_click = on_click  # CALLBACK ADDED

    @property
    def x(self):
        return self.rect.x

    @x.setter
    def x(self, value):
        self.rect.x = value

    @property
    def y(self):
        return self.rect.y

    @y.setter
    def y(self, value):
        self.rect.y = value

    def draw(self, screen):
        color = (
            tuple(min(255, c + 20) for c in self.color) if self.hover else self.color
        )
        pygame.draw.rect(screen, color, self.rect, border_radius=5)

        text_surface = pygame.font.Font(None, self.name_size).render(
            self.text, True, self.name_color
        )
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                # CALLBACK ADDED
                if self.on_click:
                    self.on_click()
                return True
        return False


class Label(Component):
    text_color = (255, 255, 255)
    small_size = 24
    medium_size = 28
    large_size = 36

    def __init__(
        self, x, y, text, component_id, font_size="small", color=(255, 255, 255)
    ):
        if font_size == "large":
            height = self.large_size
        elif font_size == "medium":
            height = self.medium_size
        else:
            height = self.small_size

        super().__init__(x, y, 0, height, component_id)
        self.text = text
        self.color = color
        self.font_size = height

    def draw(self, screen):
        text_surface = pygame.font.Font(None, self.font_size).render(
            self.text, True, self.color
        )
        screen.blit(text_surface, (self.x, self.y))

    def handle_event(self, event):
        pass
