import pytest
import pygame
import pygame_gui
from display.canvas import Canvas


@pytest.fixture
def pygame_setup():
    pygame.init()
    pygame.display.set_mode((100, 100))
    yield
    pygame.quit()


@pytest.fixture
def manager(pygame_setup):
    return pygame_gui.UIManager((800, 600))


class TestCanvas:

    def test_canvas_initialization(self, manager):

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 700, 700), manager=manager
        )

        assert canvas.rect.x == 0
        assert canvas.rect.y == 0
        assert canvas.rect.width == 700
        assert canvas.rect.height == 700
        assert canvas.image is not None
        assert canvas.on_draw is None
        assert canvas.on_click is None

    def test_canvas_with_callbacks(self, manager):
        def draw_func(surface):
            pass

        def click_func(x, y, button):
            pass

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 500, 500),
            manager=manager,
            on_draw=draw_func,
            on_click=click_func,
        )

        assert canvas.on_draw == draw_func
        assert canvas.on_click == click_func

    def test_draw_content_clears_to_white(self, manager):

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 100, 100), manager=manager
        )

        canvas.draw_content()

        pixel = canvas.image.get_at((50, 50))
        assert pixel[:3] == (255, 255, 255)

    def test_draw_content_calls_on_draw_callback(self, manager):
        called = []

        def draw_func(surface):
            called.append(True)
            assert isinstance(surface, pygame.Surface)

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 200, 200),
            manager=manager,
            on_draw=draw_func,
        )

        canvas.draw_content()

        assert len(called) == 1

    def test_draw_content_without_callback(self, manager):

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 200, 200), manager=manager
        )

        canvas.draw_content()

    def test_click_inside_canvas_triggers_callback(self, manager):
        click_data = []

        def click_func(x, y, button):
            click_data.append((x, y, button))

        canvas = Canvas(
            relative_rect=pygame.Rect(100, 50, 400, 300),
            manager=manager,
            on_click=click_func,
        )

        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(200, 100), button=1
        )

        result = canvas.process_event(event)

        assert result is True
        assert len(click_data) == 1
        assert click_data[0] == (100, 50, 1)

    def test_click_outside_canvas_does_not_trigger_callback(self, manager):
        click_data = []

        def click_func(x, y, button):
            click_data.append((x, y, button))

        canvas = Canvas(
            relative_rect=pygame.Rect(100, 100, 200, 200),
            manager=manager,
            on_click=click_func,
        )

        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(50, 50), button=1
        )

        result = canvas.process_event(event)

        assert result is False
        assert len(click_data) == 0

    def test_coordinate_transformation_to_local(self, manager):
        click_data = []

        def click_func(x, y, button):
            click_data.append((x, y, button))

        canvas = Canvas(
            relative_rect=pygame.Rect(200, 150, 300, 300),
            manager=manager,
            on_click=click_func,
        )

        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(250, 200), button=1
        )

        canvas.process_event(event)

        assert click_data[0][0] == 50
        assert click_data[0][1] == 50

    def test_left_click_detected(self, manager):
        click_data = []

        def click_func(x, y, button):
            click_data.append(button)

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 500, 500),
            manager=manager,
            on_click=click_func,
        )

        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(100, 100), button=1
        )

        canvas.process_event(event)

        assert click_data[0] == 1

    def test_right_click_detected(self, manager):
        click_data = []

        def click_func(x, y, button):
            click_data.append(button)

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 500, 500),
            manager=manager,
            on_click=click_func,
        )

        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(100, 100), button=3
        )

        canvas.process_event(event)

        assert click_data[0] == 3

    def test_click_without_callback_doesnt_crash(self, manager):

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 500, 500), manager=manager
        )

        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(100, 100), button=1
        )

        result = canvas.process_event(event)

        assert result is True

    def test_non_click_event_returns_false(self, manager):

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 500, 500), manager=manager
        )

        event = pygame.event.Event(pygame.MOUSEMOTION, pos=(100, 100))

        result = canvas.process_event(event)

        assert result is False

    def test_canvas_image_size_matches_rect(self, manager):
        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 640, 480), manager=manager
        )

        assert canvas.image.get_width() == 640
        assert canvas.image.get_height() == 480

    def test_multiple_draw_content_calls(self, manager):
        draw_count = []

        def draw_func(surface):
            draw_count.append(1)

        canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 200, 200),
            manager=manager,
            on_draw=draw_func,
        )

        canvas.draw_content()
        canvas.draw_content()
        canvas.draw_content()

        assert len(draw_count) == 3
