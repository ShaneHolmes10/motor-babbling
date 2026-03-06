import pygame
import pygame_gui
from display.canvas import Canvas
from display.control_panel import ControlPanelBuilder

pygame.init()

CANVAS_WIDTH = 700
PANEL_WIDTH = 300
WIDTH = CANVAS_WIDTH + PANEL_WIDTH
HEIGHT = 700

screen = pygame.display.set_mode((WIDTH, HEIGHT))
manager = pygame_gui.UIManager((WIDTH, HEIGHT))

# Robot state
target = [0, 0]


# Canvas callbacks
def handle_click(x, y, button):
    global target
    if button == 1:  # Left click
        target = [x, y]  # Add new object
        print(f"Added target at {x}, {y}")
    elif button == 3:  # Right click
        target = [0, 0]  # Clear all objects
        print("Cleared all targets")


def draw_scene(surface):
    surface.fill((255, 255, 255))

    # Draw all objects from shared state
    pygame.draw.circle(surface, (255, 0, 0), target, 10)
    # Draw cross-hairs
    pygame.draw.line(
        surface,
        (255, 0, 0),
        (target[0] - 15, target[1]),
        (target[0] + 15, target[1]),
        2,
    )
    pygame.draw.line(
        surface,
        (255, 0, 0),
        (target[0], target[1] - 15),
        (target[0], target[1] + 15),
        2,
    )


# Control panel callbacks


def on_joint1_change(value):
    target[0] += 0.01 * value
    print(f"x component: {value:.1f}°")


def on_joint2_change(value):
    target[1] += 0.01 * value
    print(f"y component: {value:.1f}°")


def on_reset():
    print("Reset!")
    control_panel.set_slider_value("joint1", 0)
    control_panel.set_slider_value("joint2", 0)


# Create canvas

canvas = Canvas(
    relative_rect=pygame.Rect(0, 0, 700, 700),
    manager=manager,
    on_draw=draw_scene,
    on_click=handle_click,
)

# Build control panel using builder

builder = ControlPanelBuilder(
    x=CANVAS_WIDTH, y=0, width=PANEL_WIDTH, height=HEIGHT, manager=manager
)

control_panel = (
    builder.add_section("Joint Controls", height=200)
    .add_slider(
        "joint1",
        -180,
        180,
        label="Joint 1",
        initial=0,
        on_change=on_joint1_change,
    )
    .add_slider(
        "joint2",
        -180,
        180,
        label="Joint 2",
        initial=0,
        on_change=on_joint2_change,
    )
    .add_slider(
        "joint3",
        -180,
        180,
        label="Joint 3",
        initial=0,
        on_change=on_joint2_change,
    )
    .add_slider(
        "joint4",
        -180,
        180,
        label="Joint 4",
        initial=0,
        on_change=on_joint2_change,
    )
    .end_section()
    .add_section("Actions", height=120)
    .add_button("reset", "Reset to Home", on_click=on_reset)
    .end_section()
    .build()
)


clock = pygame.time.Clock()
running = True


while running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        manager.process_events(event)

    manager.update(time_delta)

    screen.fill((42, 42, 42))

    canvas.draw_content()

    manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
