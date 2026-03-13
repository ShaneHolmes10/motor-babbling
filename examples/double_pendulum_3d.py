import pybullet as p
import pybullet_data
import numpy as np
import time

# Initialize PyBullet with GUI
physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)

# Create double pendulum
link_length = 0.5
link_mass = 1.0

collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.05,
    height=link_length,
    collisionFramePosition=[0, 0, -link_length/2]  
)

visual_shape = p.createVisualShape(
    shapeType=p.GEOM_CYLINDER,
    radius=0.05,
    length=link_length,
    rgbaColor=[0.8, 0.3, 0.3, 1.0],
    visualFramePosition=[0, 0, -link_length/2] 
)

pendulum = p.createMultiBody(
    baseMass=0,
    basePosition=[0, 0, 0],
    linkMasses=[link_mass, link_mass],
    linkCollisionShapeIndices=[collision_shape, collision_shape],
    linkVisualShapeIndices=[visual_shape, visual_shape],
    linkPositions=[[0, 0, 0], [0, 0, -link_length]],  # Joint positions relative to parent
    linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
    linkInertialFramePositions=[[0, 0, -link_length/2], [0, 0, -link_length/2]],  # COM at center
    linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
    linkParentIndices=[0, 1],
    linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
    linkJointAxis=[[0, 1, 0], [0, 1, 0]]
)

# Set initial angles (start high so it falls)
p.resetJointState(pendulum, 0, np.pi/2, 0)
p.resetJointState(pendulum, 1, np.pi/2, 0)

# Disable motors so joints can move freely
for joint_idx in [0, 1]:
    p.setJointMotorControl2(
        pendulum,
        joint_idx,
        p.VELOCITY_CONTROL,
        force=0
    )
    p.changeDynamics(
        pendulum,
        joint_idx,
        linearDamping=0.01,
        angularDamping=0.01,
        jointDamping=0.01
    )

# Set camera position for better view
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=0,
    cameraPitch=-20,
    cameraTargetPosition=[0, 0, -0.5]
)

print("Double Pendulum Simulation")
print("Close the window to exit")

# Simulation loop
while True:
    p.stepSimulation()
    time.sleep(1.0/240.0)