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

robot = p.loadURDF('rl-quadruped/mm-walker/mm_walker/envs/urdf/robot_simple.urdf', [0, 0, 1], flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
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
    links2 = {1:"front_left",
                3:"rear_left",
                5:"front_right",
                7:"rear_right"}
    for i in links2.keys():
        cp = p.getContactPoints(robot,robot,i)
        for point in cp:
            print(f'Collision point {links2[i]}: {point}')
    time.sleep(1./240.)
