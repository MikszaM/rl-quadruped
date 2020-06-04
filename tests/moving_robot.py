import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setGravity(0, 0, -100)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

angles = []
for i in range(8):
    angles.append(p.addUserDebugParameter(f'servo_{i}', -3, 3, 0))

robot = p.loadURDF('simulation/robot.urdf', [0, 0, 1])
number_of_joints = p.getNumJoints(robot)
for joint_number in range(number_of_joints):
    info = p.getJointInfo(robot, joint_number)
    print(info)

print(f"Angles: {angles}")
user_angles = []
while True:
    for i in range(8):
        user_angle = p.readUserDebugParameter(angles[i])
        p.setJointMotorControl2(robot, i,
                            p.POSITION_CONTROL,
                            targetPosition=user_angle)
    
    p.stepSimulation()
    time.sleep(1./240.)
