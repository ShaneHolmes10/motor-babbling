import pygame
import pygame_gui
from display.canvas import Canvas

pygame.init()

screen = pygame.display.set_mode((700, 700))
manager = pygame_gui.UIManager((700, 700))


def draw_object(surface: pygame.Surface):
    pygame.draw.circle(surface, (255, 0, 0), (100, 100), 50)


# Create canvas
# fmt: off
canvas = Canvas(
    relative_rect=pygame.Rect(0, 0, 700, 700),
    manager=manager,
    on_draw=draw_object
)
# fmt: on

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
