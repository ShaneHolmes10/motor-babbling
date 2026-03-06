import pygame
import pygame_gui
from display.canvas import Canvas
from display.control_panel import ControlPanelBuilder


class RobotApp:
    """
    Main application class for the robot arm controller.
    Manages the GUI, robot state, and coordinates between canvas and control
    panel.
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

        # Create canvas
        self.canvas = Canvas(
            relative_rect=pygame.Rect(0, 0, 700, 700),
            manager=self.manager,
            on_draw=self.draw_scene,
            on_click=self.handle_click,
        )

        # Build control panel using builder
        builder = ControlPanelBuilder(
            x=CANVAS_WIDTH,
            y=0,
            width=PANEL_WIDTH,
            height=HEIGHT,
            manager=self.manager,
        )

        self.control_panel = (
            builder.add_section("Joint Controls", height=200)
            .add_slider(
                "joint1",
                -180,
                180,
                label="Joint 1",
                initial=0,
                on_change=self.on_joint1_change,
            )
            .add_slider(
                "joint2",
                -180,
                180,
                label="Joint 2",
                initial=0,
                on_change=self.on_joint2_change,
            )
            .add_slider(
                "joint3",
                -180,
                180,
                label="Joint 3",
                initial=0,
                on_change=self.on_joint2_change,
            )
            .add_slider(
                "joint4",
                -180,
                180,
                label="Joint 4",
                initial=0,
                on_change=self.on_joint2_change,
            )
            .end_section()
            .add_section("Actions", height=120)
            .add_button("reset", "Reset to Home", on_click=self.on_reset)
            .end_section()
            .build()
        )

        # Robot state
        self.model = [0, 0]

    # Canvas callbacks
    def handle_click(self, x, y, button):
        """
        Handle canvas click events.

        Args:
            x: X coordinate in canvas-local space
            y: Y coordinate in canvas-local space
            button: Mouse button (1=left, 3=right)
        """

        if button == 1:  # Left click
            self.model = [x, y]  # Add new object
            print(f"Added target at {x}, {y}")
        elif button == 3:  # Right click
            self.model = [0, 0]  # Clear all objects
            print("Cleared all targets")

    def draw_scene(self, surface):
        """
        Draw the robot visualization on the canvas.
        Called by Canvas.draw_content() each frame.

        Args:
            surface: pygame.Surface to draw on
        """

        surface.fill((255, 255, 255))

        # Draw all objects from shared state
        pygame.draw.circle(surface, (255, 0, 0), self.model, 10)
        pygame.draw.line(
            surface,
            (255, 0, 0),
            (self.model[0] - 15, self.model[1]),
            (self.model[0] + 15, self.model[1]),
            2,
        )
        pygame.draw.line(
            surface,
            (255, 0, 0),
            (self.model[0], self.model[1] - 15),
            (self.model[0], self.model[1] + 15),
            2,
        )

    # Control panel callbacks
    def on_joint1_change(self, value):
        """
        Handle Joint 1 slider changes.

        Args:
            value: New slider value in degrees
        """

        self.model[0] += 0.01 * value
        print(f"x component: {value:.1f}°")

    def on_joint2_change(self, value):
        """
        Handle Joint 2 slider changes.

        Args:
            value: New slider value in degrees
        """

        self.model[1] += 0.01 * value
        print(f"y component: {value:.1f}°")

    def on_reset(self):
        """Handle reset button click - reset all joint sliders to zero."""

        print("Reset!")
        self.control_panel.set_slider_value("joint1", 0)
        self.control_panel.set_slider_value("joint2", 0)

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
