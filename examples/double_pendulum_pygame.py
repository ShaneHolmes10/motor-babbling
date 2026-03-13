import pygame
import pybullet as p
import pybullet_data
import numpy as np

pygame.init()

WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Double Pendulum - Pygame Visualization")

# Initialize PyBullet in DIRECT mode (headless - no 3D GUI)
physics_client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

# Create double pendulum
link_length = 0.5
link_mass = 1.0

collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.05,
    height=link_length,
    collisionFramePosition=[0, 0, -link_length / 2],
)

visual_shape = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.05,
    length=link_length,
    rgbaColor=[0.8, 0.3, 0.3, 1.0],
    visualFramePosition=[0, 0, -link_length / 2],
)

pendulum = p.createMultiBody(
    baseMass=0,
    basePosition=[0, 0, 0],
    linkMasses=[link_mass, link_mass, 0.0],  # Third link has zero mass
    linkCollisionShapeIndices=[
        collision_shape,
        collision_shape,
        -1,
    ],  # No collision for dummy
    linkVisualShapeIndices=[
        visual_shape,
        visual_shape,
        -1,
    ],  # No visual for dummy
    linkPositions=[
        [0, 0, 0],
        [0, 0, -link_length],
        [0, 0, -link_length],
    ],  # Dummy at end of link2
    linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
    linkInertialFramePositions=[
        [0, 0, -link_length / 2],
        [0, 0, -link_length / 2],
        [0, 0, 0],
    ],
    linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
    linkParentIndices=[0, 1, 2],  # Dummy connects to link2
    linkJointTypes=[
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_FIXED,
    ],  # Fixed joint = no movement
    linkJointAxis=[[0, 1, 0], [0, 1, 0], [0, 1, 0]],
)

# Set initial angles
p.resetJointState(pendulum, 0, np.pi / 2, 0)
p.resetJointState(pendulum, 1, np.pi / 2, 0)

# Disable motors
for joint_idx in [0, 1]:
    p.setJointMotorControl2(pendulum, joint_idx, p.VELOCITY_CONTROL, force=0)
    p.changeDynamics(
        pendulum,
        joint_idx,
        linearDamping=0.01,
        angularDamping=0.01,
        jointDamping=0.01,
    )

# Rendering parameters
scale = 200.0
center_x = WIDTH // 2
center_y = 100

clock = pygame.time.Clock()
running = True
frame_count = 0

print("Double Pendulum Pygame Visualization")
print("Close the window to exit")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Step physics simulation
    p.stepSimulation()

    # Get link states from PyBullet
    link1_state = p.getLinkState(pendulum, 0)
    link2_state = p.getLinkState(pendulum, 1)
    link3_state = p.getLinkState(pendulum, 2)

    # Get CENTER OF MASS positions (middle of each cylinder)
    base_pos = [0, 0, 0]

    # Also get joint positions to draw joints
    link1_joint_pos = link1_state[4]
    link2_joint_pos = link2_state[4]  # Should be at base
    link3_joint_pos = link3_state[4]  # Joint between link1 and link2

    # Debug: print positions every 60 frames
    frame_count += 1
    if frame_count % 60 == 0:
        print(f"\nFrame {frame_count}:")

    # Convert 3D positions to 2D screen coordinates (X-Z plane projection)

    link1_screen = (
        int(center_x + link1_joint_pos[0] * scale),
        int(center_y - link1_joint_pos[2] * scale),
    )
    link2_screen = (
        int(center_x + link2_joint_pos[0] * scale),
        int(center_y - link2_joint_pos[2] * scale),
    )
    link3_screen = (
        int(center_x + link3_joint_pos[0] * scale),
        int(center_y - link3_joint_pos[2] * scale),
    )

    # Clear screen
    screen.fill((255, 255, 255))

    # Draw links through their centers of mass
    pygame.draw.line(screen, (200, 80, 80), link1_screen, link2_screen, 10)
    pygame.draw.line(screen, (200, 80, 80), link2_screen, link3_screen, 10)

    # Draw joints as circles
    pygame.draw.circle(screen, (50, 50, 50), link1_screen, 8)
    pygame.draw.circle(screen, (100, 100, 200), link2_screen, 8)
    pygame.draw.circle(screen, (100, 200, 100), link3_screen, 8)

    pygame.display.flip()
    clock.tick(60)

p.disconnect()
pygame.quit()
