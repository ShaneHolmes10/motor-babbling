import pygame
import pygame_gui
from display.canvas import Canvas
from display.control_panel import ControlPanelBuilder


class RobotApp:
    """
    Demo application showing control panel and canvas interaction.
    Controls two colored balls with sliders and displays their coordinates.
    """

    def __init__(self):
        """Initialize the application with GUI components and initial state."""

        pygame.init()

        CANVAS_WIDTH = 700
        PANEL_WIDTH = 300
        WIDTH = CANVAS_WIDTH + PANEL_WIDTH
        HEIGHT = 700

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.manager = pygame_gui.UIManager((WIDTH, HEIGHT))

        # Ball positions (start in center of canvas)
        self.red_ball = [350, 350]
        self.green_ball = [350, 350]

        # Click markers
        self.click_markers = []

        # Create canvas element for visualization
        self.canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, CANVAS_WIDTH, HEIGHT),
            manager=self.manager,
            on_draw=self.draw_scene,
            on_click=self.handle_click,
        )

        # Build control panel using builder pattern
        builder = ControlPanelBuilder(
            x=CANVAS_WIDTH,
            y=0,
            width=PANEL_WIDTH,
            height=HEIGHT,
            manager=self.manager,
        )

        self.control_panel = (
            builder.add_section("Red Ball Controls", height=220)
            .add_slider(
                "red_x",
                0,
                700,
                label="Red X",
                initial=350,
                on_change=self.on_red_x_change,
            )
            .add_slider(
                "red_y",
                0,
                700,
                label="Red Y",
                initial=350,
                on_change=self.on_red_y_change,
            )
            .add_label("red_pos", "Red: (350, 350)")
            .end_section()
            .add_section("Green Ball Controls", height=220)
            .add_slider(
                "green_x",
                0,
                700,
                label="Green X",
                initial=350,
                on_change=self.on_green_x_change,
            )
            .add_slider(
                "green_y",
                0,
                700,
                label="Green Y",
                initial=350,
                on_change=self.on_green_y_change,
            )
            .add_label("green_pos", "Green: (350, 350)")
            .end_section()
            .add_section("Actions", height=200)
            .add_button("reset", "Reset to Center", on_click=self.on_reset)
            .add_button(
                "clear_markers",
                "Clear Markers",
                on_click=self.on_clear_markers,
            )
            .end_section()
            .build()
        )

    def handle_click(self, x, y, button):
        """
        Handle canvas click events.

        Args:
            x: X coordinate in canvas-local space
            y: Y coordinate in canvas-local space
            button: Mouse button (1=left, 3=right)
        """
        if button == 1:
            self.click_markers.append((x, y))
            print(f"Added click marker at ({x}, {y})")

    def draw_scene(self, surface):
        """
        Draw the visualization on the canvas.

        Args:
            surface: pygame.Surface to draw on
        """
        surface.fill((255, 255, 255))

        # Draw grid
        gray = (200, 200, 200)
        for i in range(0, 700, 50):
            pygame.draw.line(surface, gray, (i, 0), (i, 700), 1)
            pygame.draw.line(surface, gray, (0, i), (700, i), 1)

        # Draw red ball
        pygame.draw.circle(surface, (255, 0, 0), self.red_ball, 20)

        # Draw green ball
        pygame.draw.circle(surface, (0, 255, 0), self.green_ball, 20)

        # Draw click markers (small blue dots)
        for marker in self.click_markers:
            pygame.draw.circle(surface, (0, 0, 255), marker, 5)

    def on_red_x_change(self, value):
        """Update red ball X position and label."""
        self.red_ball[0] = int(value)
        self._update_red_label()

    def on_red_y_change(self, value):
        """Update red ball Y position and label."""
        self.red_ball[1] = int(value)
        self._update_red_label()

    def on_green_x_change(self, value):
        """Update green ball X position and label."""
        self.green_ball[0] = int(value)
        self._update_green_label()

    def on_green_y_change(self, value):
        """Update green ball Y position and label."""
        self.green_ball[1] = int(value)
        self._update_green_label()

    def _update_red_label(self):
        """Update the red ball position label."""
        label = self.control_panel.get_widget("red_pos")
        if label:
            label.set_text(f"Red: ({self.red_ball[0]}, {self.red_ball[1]})")

    def _update_green_label(self):
        """Update the green ball position label."""
        label = self.control_panel.get_widget("green_pos")
        if label:
            label.set_text(
                f"Green: ({self.green_ball[0]}, {self.green_ball[1]})"
            )

    def on_reset(self):
        """Reset both balls and all sliders to center position."""
        print("Reset to center!")

        # Reset sliders
        self.control_panel.set_slider_value("red_x", 350)
        self.control_panel.set_slider_value("red_y", 350)
        self.control_panel.set_slider_value("green_x", 350)
        self.control_panel.set_slider_value("green_y", 350)

        # Reset ball positions
        self.red_ball = [350, 350]
        self.green_ball = [350, 350]

        # Update labels
        self._update_red_label()
        self._update_green_label()

    def on_clear_markers(self):
        """Clear all click markers from canvas."""
        print("Cleared markers!")
        self.click_markers.clear()

    def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        running = True

        while running:
            time_delta = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.manager.process_events(event)

            self.manager.update(time_delta)

            self.screen.fill((42, 42, 42))

            self.canvas.draw_content()

            self.manager.draw_ui(self.screen)

            pygame.display.flip()

        pygame.quit()
