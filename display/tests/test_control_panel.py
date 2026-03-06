import pytest
import pygame
import pygame_gui
from display.control_panel import ControlPanel, ControlPanelBuilder


@pytest.fixture
def pygame_setup():
    pygame.init()
    pygame.display.set_mode((100, 100))
    yield
    pygame.quit()


@pytest.fixture
def manager(pygame_setup):
    return pygame_gui.UIManager((1000, 700))


class TestControlPanelBuilder:

    def test_builder_initialization(self, manager):
        builder = ControlPanelBuilder(
            x=700, y=0, width=300, height=700, manager=manager
        )

        assert builder.x == 700
        assert builder.y == 0
        assert builder.width == 300
        assert builder.height == 700
        assert builder.manager == manager
        assert builder.current_y == 0
        assert len(builder.widgets) == 0
        assert len(builder.callbacks) == 0

    def test_add_section(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        result = builder.add_section("Test Section", height=200)

        assert result is builder
        assert builder.current_section is not None
        assert builder.current_section["current_y"] == 40

    def test_add_section_without_title(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        builder.add_section(None, height=200)

        assert builder.current_section["current_y"] == 10

    def test_add_slider_requires_section(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )

        with pytest.raises(ValueError, match="Must call add_section first"):
            builder.add_slider("joint1", -180, 180)

    def test_add_slider(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        result = builder.add_section("Joints", height=200).add_slider(
            "joint1", -180, 180, label="Joint 1", initial=0
        )

        assert result is builder
        assert "joint1" in builder.widgets
        assert isinstance(
            builder.widgets["joint1"], pygame_gui.elements.UIHorizontalSlider
        )

    def test_add_slider_with_callback(self, manager):
        called = []

        def on_change(value):
            called.append(value)

        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        builder.add_section("Joints", height=200)
        builder.add_slider("joint1", -180, 180, on_change=on_change)

        assert "joint1_change" in builder.callbacks
        assert builder.callbacks["joint1_change"] == on_change

    def test_add_button_requires_section(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )

        with pytest.raises(ValueError, match="Must call add_section first"):
            builder.add_button("reset", "Reset")

    def test_add_button(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        result = builder.add_section("Actions", height=150).add_button(
            "reset", "Reset to Home"
        )

        assert result is builder
        assert "reset" in builder.widgets
        assert isinstance(
            builder.widgets["reset"], pygame_gui.elements.UIButton
        )

    def test_add_button_with_callback(self, manager):
        called = []

        def on_click():
            called.append(True)

        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        builder.add_section("Actions", height=150)
        builder.add_button("reset", "Reset", on_click=on_click)

        assert "reset_click" in builder.callbacks
        assert builder.callbacks["reset_click"] == on_click

    def test_add_label(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        result = builder.add_section("Info", height=100).add_label(
            "status", "Ready"
        )

        assert result is builder
        assert "status" in builder.widgets

    def test_add_text_box(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        result = builder.add_section("Info", height=150).add_text_box(
            "details", "<b>Info:</b><br>Text here", height=80
        )

        assert result is builder
        assert "details" in builder.widgets

    def test_end_section_requires_active_section(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )

        with pytest.raises(ValueError, match="No section to end"):
            builder.end_section()

    def test_end_section_clears_current_section(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        builder.add_section("Test", height=200)
        builder.end_section()

        assert builder.current_section is None

    def test_method_chaining(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )

        result = (
            builder.add_section("Joints", height=200)
            .add_slider("joint1", -180, 180)
            .add_slider("joint2", -180, 180)
            .end_section()
            .add_section("Actions", height=150)
            .add_button("reset", "Reset")
            .end_section()
        )

        assert result is builder

    def test_build_returns_control_panel(self, manager):
        builder = ControlPanelBuilder(
            x=700, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Test", height=100)
            .add_slider("s1", 0, 100)
            .end_section()
            .build()
        )

        assert isinstance(panel, ControlPanel)

    def test_complex_panel_structure(self, manager):
        builder = ControlPanelBuilder(
            x=700, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Joints", height=200)
            .add_slider("joint1", -180, 180, label="Joint 1")
            .add_slider("joint2", -180, 180, label="Joint 2")
            .end_section()
            .add_section("Actions", height=150)
            .add_button("reset", "Reset")
            .add_button("random", "Random")
            .end_section()
            .build()
        )

        assert "joint1" in panel.widgets
        assert "joint2" in panel.widgets
        assert "reset" in panel.widgets
        assert "random" in panel.widgets


class TestControlPanel:

    def test_control_panel_initialization(self, manager):
        widgets = {}
        callbacks = {}
        panel = ControlPanel(
            relative_rect=pygame.Rect(700, 0, 300, 700),
            manager=manager,
            widgets=widgets,
            callbacks=callbacks,
        )

        assert panel.widgets == widgets
        assert panel.callbacks == callbacks

    def test_get_widget(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Test", height=100)
            .add_slider("joint1", 0, 100)
            .end_section()
            .build()
        )

        slider = panel.get_widget("joint1")
        assert slider is not None
        assert slider == panel.widgets["joint1"]

    def test_get_nonexistent_widget(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = builder.add_section("Test", 100).end_section().build()

        widget = panel.get_widget("nonexistent")
        assert widget is None

    def test_set_slider_value(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Test", height=100)
            .add_slider("joint1", -180, 180, initial=0)
            .end_section()
            .build()
        )

        panel.set_slider_value("joint1", 90)

        slider = panel.get_widget("joint1")
        assert slider.get_current_value() == 90

    def test_set_nonexistent_slider_doesnt_crash(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = builder.add_section("Test", 100).end_section().build()

        panel.set_slider_value("nonexistent", 50)

    def test_slider_callback_fires(self, manager):
        called_values = []

        def on_change(value):
            called_values.append(value)

        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Test", height=100)
            .add_slider("joint1", 0, 100, on_change=on_change)
            .end_section()
            .build()
        )

        slider = panel.get_widget("joint1")
        event = pygame.event.Event(
            pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=slider
        )

        panel.process_event(event)

        assert len(called_values) == 1

    def test_button_callback_fires(self, manager):
        called = []

        def on_click():
            called.append(True)

        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Actions", height=100)
            .add_button("reset", "Reset", on_click=on_click)
            .end_section()
            .build()
        )

        button = panel.get_widget("reset")
        event = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED, ui_element=button
        )

        panel.process_event(event)

        assert len(called) == 1

    def test_event_without_callback_doesnt_crash(self, manager):
        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Test", height=100)
            .add_slider("joint1", 0, 100)
            .end_section()
            .build()
        )

        slider = panel.get_widget("joint1")
        event = pygame.event.Event(
            pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=slider
        )

        panel.process_event(event)

    def test_multiple_sliders_with_callbacks(self, manager):
        values = {}

        def on_j1_change(value):
            values["j1"] = value

        def on_j2_change(value):
            values["j2"] = value

        builder = ControlPanelBuilder(
            x=0, y=0, width=300, height=700, manager=manager
        )
        panel = (
            builder.add_section("Joints", height=200)
            .add_slider("joint1", -180, 180, on_change=on_j1_change)
            .add_slider("joint2", -180, 180, on_change=on_j2_change)
            .end_section()
            .build()
        )

        j1_slider = panel.get_widget("joint1")
        j2_slider = panel.get_widget("joint2")

        event1 = pygame.event.Event(
            pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=j1_slider
        )
        panel.process_event(event1)

        event2 = pygame.event.Event(
            pygame_gui.UI_HORIZONTAL_SLIDER_MOVED, ui_element=j2_slider
        )
        panel.process_event(event2)

        assert "j1" in values
        assert "j2" in values
