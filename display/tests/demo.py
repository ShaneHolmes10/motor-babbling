import pygame
from display.control_panel import ControlPanelBuilder
import random

pygame.init()

WIDTH = 400
HEIGHT = 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Control Panel Demo")

WHITE = (255, 255, 255)
GRAY = (200, 200, 200)


def main():
    clock = pygame.time.Clock()
    running = True

    # Store current joint values
    joint_values = {"joint1": 0, "joint2": 45, "joint3": -30}

    # Define callback functions
    def on_joint1_change(value):
        joint_values["joint1"] = value
        print(f"Joint 1: {value:.1f}°")

    def on_joint2_change(value):
        joint_values["joint2"] = value
        print(f"Joint 2: {value:.1f}°")

    def on_joint3_change(value):
        joint_values["joint3"] = value
        print(f"Joint 3: {value:.1f}°")

    def on_reset_click():
        print("\nReset button clicked!")
        panel.set_slider_value("joint1", 0)
        panel.set_slider_value("joint2", 0)
        panel.set_slider_value("joint3", 0)
        print("  All joints reset to 0°")

    def on_random_click():
        print("\nRandom button clicked!")
        j1 = random.uniform(-180, 180)
        j2 = random.uniform(-180, 180)
        j3 = random.uniform(-90, 90)

        panel.set_slider_value("joint1", j1)
        panel.set_slider_value("joint2", j2)
        panel.set_slider_value("joint3", j3)

        print(f"  Joint 1: {j1:.1f}°")
        print(f"  Joint 2: {j2:.1f}°")
        print(f"  Joint 3: {j3:.1f}°")

    def on_save_click():
        speed = panel.get_slider_value("speed")
        precision = panel.get_slider_value("precision")

        print("\nSave button clicked!")
        print("  Current Configuration:")
        print(f"    Joint 1: {joint_values['joint1']:.1f}°")
        print(f"    Joint 2: {joint_values['joint2']:.1f}°")
        print(f"    Joint 3: {joint_values['joint3']:.1f}°")
        print(f"    Speed: {speed:.1f}%")
        print(f"    Precision: {precision:.1f}")

    # Build panel with callbacks
    builder = ControlPanelBuilder(x=0, y=0, width=WIDTH, height=HEIGHT)

    panel = (
        builder.add_label(
            "title", "Robot Control Panel", font_size="large", color=(74, 158, 255)
        )
        .add_spacing(20)
        .begin_container("joints", height=250, title="Joint Controls")
        .add_slider(
            "joint1",
            -180,
            180,
            initial=0,
            label="Joint 1",
            on_value_change=on_joint1_change,
        )
        .add_slider(
            "joint2",
            -180,
            180,
            initial=45,
            label="Joint 2",
            on_value_change=on_joint2_change,
        )
        .add_slider(
            "joint3",
            -90,
            90,
            initial=-30,
            label="Joint 3",
            on_value_change=on_joint3_change,
        )
        .add_slider(
            "joint4",
            -90,
            90,
            initial=-30,
            label="Joint 3",
            on_value_change=on_joint3_change,
        )
        .end_container()
        .begin_container("actions", height=180, title="Actions")
        .add_button(
            "reset", "Reset to Home", color=(74, 158, 255), on_click=on_reset_click
        )
        .add_button(
            "random", "Random Pose", color=(102, 102, 102), on_click=on_random_click
        )
        .add_button(
            "save", "Save Position", color=(76, 175, 80), on_click=on_save_click
        )
        .end_container()
        .begin_container("settings", height=150, title="Settings")
        .add_slider("speed", 0, 100, initial=50, label="Speed")
        .add_slider("precision", 1, 10, initial=5, label="Precision")
        .end_container()
        .build()
    )

    print("Control Panel Demo Started!")
    print("=" * 50)
    print("Try the following interactions:")
    print("  - Drag the sliders to change values")
    print("  - Scroll inside containers with mouse wheel")
    print("  - Click the buttons to trigger actions")
    print("  - Drag the scrollbar handles for precise scrolling")
    print("=" * 50)

    # Clean event-driven main loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                # Components handle events and call callbacks automatically
                panel.handle_event(event)

        screen.fill(GRAY)
        panel.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\nDemo closed.")


if __name__ == "__main__":
    main()
