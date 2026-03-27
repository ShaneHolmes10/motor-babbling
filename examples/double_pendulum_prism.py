import pybullet as p
import numpy as np
import time

physics_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)

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

robot = p.createMultiBody(
    baseMass=0,
    basePosition=[0, 0, 0],
    linkMasses=[link_mass, link_mass],
    linkCollisionShapeIndices=[collision_shape, collision_shape],
    linkVisualShapeIndices=[visual_shape, visual_shape],
    linkPositions=[[0, 0, 0], [0, 0, -link_length]],
    linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
    linkInertialFramePositions=[[0, 0, -link_length/2], [0, 0, -link_length/2]],
    linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
    linkParentIndices=[0, 1],  # FIXED: link2 attaches to link1
    linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
    linkJointAxis=[[0, 1, 0], [0, 1, 0]]
)

# Set joint limits
'''
p.changeDynamics(robot, 0, 
                jointLowerLimit=0.0,    # Min extension
                jointUpperLimit=0.5,    # Max extension (0.5m)
                jointLimitForce=1000)
'''

# Start extended
p.resetJointState(robot, 0, 0.3, 0)
p.resetJointState(robot, 1, np.pi/4, 0)

# Disable motors for free motion
for joint_idx in [0, 1]:
    p.setJointMotorControl2(robot, joint_idx, p.VELOCITY_CONTROL, force=0)

p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=0,
    cameraPitch=-20,
    cameraTargetPosition=[0, 0, -0.5]
)

print("Prismatic + Revolute Robot")
print("Joint 0: Prismatic (0 to 0.5m)")
print("Joint 1: Revolute (-180 to +180 deg)")

while True:
    p.stepSimulation()
    time.sleep(1.0/240.0)