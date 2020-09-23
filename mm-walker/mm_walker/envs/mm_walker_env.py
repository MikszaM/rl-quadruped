import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import os
import time
import numpy as np
from PIL import Image

class MMWalkerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render = False, logDir = ""):
        self.low_limits = np.array([ -0.2,   1.9,   0.2,  -1.9,   0.2,     1.9,   -0.2,  -1.9])
        self.high_limits = np.array([ 2.4,  -2.2,  -2.4,   2.2,  -2.4,    -2.2,    2.4,   2.2])
        self.action_space = spaces.Box(
            low=np.array([-1,-1,-1,-1,-1,-1,-1,-1]),
            high=np.array([1,1,1,1,1,1,1,1]),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(28,), dtype=np.float32)
        self.num_steps = 0
        self.dt = 1./2400.
        self.freq = 5
        self.maxVelocity = 5.1
        self.max_steps = 100
        self.potential = 0
        self._alive = 1
        self.isDebug = False
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0.315
        self.walk_target_x = 0
        self.walk_target_y = 1e4
        self.isRender = render
        self.logVideo = False
        self._seed()
        self.stateId = -1
        if self.isRender:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -100)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if logDir != "":
            self.logVideo = True
            self.logDir = logDir
            
        if self.logVideo and self.isRender:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.setRealTimeSimulation(False)
        self.cubeStartPos = [self.start_pos_x,self.start_pos_y,self.start_pos_z]
        self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(os.path.join(path, "urdf/robot.urdf"),
                                self.cubeStartPos,
                                self.cubeStartOrientation, 
                                flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self._observation = []
        for i in range(8):
            p.enableJointForceTorqueSensor(self.robot,i)


    def reset(self):
        if (self.stateId < 0):
            self.stateId = p.saveState()
        else:
            p.restoreState(self.stateId)
        self.num_steps = 0
        self._observation = self._calculate_observation(calculate_feet_contacts=False)
        self._calculate_progress()
        return self._observation

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]        


    def log(self, dir):
        self.logDir = dir
        self.logVideo = True
        

    def render(self, mode='human', close=False):
        self.isRender = True


    def debug(self):
        self.isDebug = True


    def step(self, action):  
        for i in range(int(1/(self.dt*self.freq))):
            if self.isRender and self.logVideo:
                cubePos, _ = p.getBasePositionAndOrientation(self.robot)
                #cubePosFinal = (cubePos[0]+1,cubePos[1],cubePos[2])
                cubePosFinal = (cubePos[0],cubePos[1],cubePos[2])
                #p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=80, cameraPitch=-55, cameraTargetPosition=cubePosFinal)
                p.resetDebugVisualizerCamera( cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=cubePosFinal)
            if self.logVideo:
                if i%(int(1/(self.dt*30)))==0:
                    visual=p.getCameraImage(1920, 1080, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                    rgba = bytes(visual[2])
                    img = Image.frombytes('RGBA', (1920, 1080), rgba)
                    img.save(self.logDir+f"{self.num_steps}_{i}.png")
            self._apply_action(action)
            p.stepSimulation()
        self.num_steps +=1
        #print(f"Episode length: {self.num_steps}")
        observation = self._calculate_observation()
        reward = self._calculate_reward()
        self.reward = reward
        done = self._calculate_done()
        if self.isDebug:
            print(f"Observation: {observation}")
            print(f"Done: {done}")
            print(f"Reward: {reward}")
        return observation, reward, done, {}


    def _apply_action(self,action):
        for i in range(8):
            p.setJointMotorControl2(self.robot, i,
                            p.POSITION_CONTROL,
                            targetPosition=self.scale_action(self.low_limits[i],self.high_limits[i],action[i]),
                            maxVelocity=self.maxVelocity)
        

    def scale_action(self,low,high,value):
        scaled = (value+1)*(high-low)/2+low
        return scaled


    def _calculate_reward(self):
        alive_coef = 1.0
        progress_coef = 1.0e1
        feet_collissions_coef = -1.0
        electricity_coef = -1e-6
        stall_torque_coef = -1.0e-10
        joints_at_limits_coef = -0.5
        alive = self._calculate_alive_bonus()
        collisions = self._calculate_feet_collisions()
        progress = self._calculate_progress()
        electricity_cost = self._calculate_electricity_cost()
        stall_torque_cost = self._calculate_stall_torque_cost()
        joints_at_limits = self.calculate_joints_at_limits()
        if self.isDebug:
            print(f"Alive: {alive_coef*alive}")
            print(f"Collisions: {feet_collissions_coef*collisions}")
            print(f"Progress: {progress_coef*progress}")
            print(f"Electricity_cost: {electricity_coef*electricity_cost}")
            print(f"Stall_torque_cost: {stall_torque_coef*stall_torque_cost}")
            print(f"Joints_at_limits: {joints_at_limits_coef*joints_at_limits}")

        reward = alive_coef*alive + progress_coef*progress + feet_collissions_coef * \
                collisions+electricity_coef*electricity_cost+stall_torque_coef * \
                stall_torque_cost + joints_at_limits_coef*joints_at_limits
        return reward

    def _calculate_stall_torque_cost(self):
        st_cost = float(np.square(self.joint_torques).mean())
        return st_cost


    def calculate_joints_at_limits(self):
        joints_at_limits = 0
        
        for i in range(8):
            if abs(self.joint_positions[i]-self.high_limits[i]) < 0.01:
                if self.isDebug:
                    print(f'Joint {i} at max limit')
                joints_at_limits += 1
            elif abs(self.joint_positions[i]-self.low_limits[i]) < 0.01:
                if self.isDebug:
                    print(f'Joint {i} at low limit')
                joints_at_limits += 1

        return joints_at_limits

    def _calculate_done(self):
        if self.num_steps>=self.max_steps:
            return True
        else:
            return self._alive < 0
        #return False


    def _calculate_feet_collisions(self):
        links = {1:"front_left",
                3:"rear_left",
                5:"front_right",
                7:"rear_right"}
        collision_points = 0
        for i in links.keys():
            cp = p.getContactPoints(self.robot,self.robot,i)
            collision_points += len(cp)
            if self.isDebug:
                for point in cp:
                    print(f'Collision point {links[i]}: {point}')
        if self.isDebug:
            print(f'Number of collisions:{collision_points}')
        return collision_points
           
    def _calculate_electricity_cost(self,):
        cost = float(np.abs(self.joint_torques * self.joint_velocities).mean())
        return cost

    def _calculate_progress(self):
        potential_old = self.potential
        self.potential = -self.walk_target_dist#/(self.dt*self.freq)
        progress = float(self.potential - potential_old)
        if self.isDebug:
             print(f'Progress, potential {progress},{self.potential}')
        return progress

    def _calculate_alive_bonus(self):
        links = {-1:"base",
            0:"upper_leg_front_left",
            2:"upper_leg_rear_left",
            4:"upper_leg_front_right",
            6:"upper_leg_rear_right"}
        ground_not_feet_contacts = 0
        for i in links.keys():
            cp = p.getContactPoints(self.robot,self.plane,i)
            ground_not_feet_contacts += len(cp)
        if ground_not_feet_contacts > 0:
            self._alive = -1
        else:
            self._alive = 1
        if self.isDebug:
            print(f'Number of CP:{ground_not_feet_contacts}, Alive: {self._alive}')
        return self._alive


    def _calculate_feet_contact(self,calculate_feet_contacts):
        feet_ground_contacts = []
        links = {1: "front_left",
                 3: "rear_left",
                 5: "front_right",
                 7: "rear_right"}
        if calculate_feet_contacts:
            for i in links.keys():
                cp = p.getContactPoints(self.robot,self.plane,i)
                feet_ground_contacts.append(len(cp))
        else:
            feet_ground_contacts = [0,0,0,0]
        return feet_ground_contacts


    def _calculate_observation(self, calculate_feet_contacts=True):
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        (vx, vy, vz), _ = p.getBaseVelocity(self.robot)
        base_velocity = np.array([vx, vy, vz])
        feet_ground_contacts = self._calculate_feet_contact(calculate_feet_contacts)
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - position[1], self.walk_target_x - position[0]])
       # print(f"Target distnce: {self.walk_target_dist}")
        print(f"Distance covered: {self.walk_target_y-self.walk_target_dist}")
        self.walk_target_theta = np.arctan2(self.walk_target_x - position[0],
                                            self.walk_target_y - position[1])
        angle_to_target = self.walk_target_theta + yaw
        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                            np.cos(-yaw), 0], [0, 0, 1]])
        vxn, vyn, vzn = np.dot(rot_speed,
                        base_velocity) 


        joint_states = p.getJointStates(self.robot, range(8))
        self.joint_positions = np.array([j[0] for j in joint_states])
        self.joint_velocities =  np.array([j[1] for j in joint_states])
        self.joint_torques =  np.array([j[3] for j in joint_states])

        height = position[2] - self.start_pos_z
        sinus = np.sin(angle_to_target)
        cosinus = np.cos(angle_to_target)
        observation_array = np.array([
            height,
            sinus,
            cosinus,
            vxn,vyn,vzn,
            roll,pitch,
        ])
        observation_array = np.concatenate((observation_array,
                                        self.joint_positions,
                                        self.joint_velocities,
                                        feet_ground_contacts))
        return observation_array
