import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import os
import time
import numpy as np


class MMWalkerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render = False):
        self.low_limits = np.array([-0.2,-2.4,-2.4,-0.2,-2.2,-1.9,-2.2,-1.2])
        self.high_limits = np.array([2.4,0.2,0.2,2.4,1.9,2.2,1.9,2.2])

        self.action_space = spaces.Box(
            low=self.low_limits,
            high=self.high_limits,
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(28,), dtype=np.float32)
        self._observation = []

        self.dt = 1./2400.
        self.freq = 1
        self.maxVelocity = 5.1
        self.potential = 0
        self._alive = 1
        self.isDebug = False
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0.315
        self.walk_target_x = 0
        self.walk_target_y = 1e3
        self.isRender = render
        self.logVideo = False
        self._seed()

    def reset(self):
        if self.isRender:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -100)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        if self.logVideo and self.isRender:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.logDir)
        cubeStartPos = [self.start_pos_x,self.start_pos_y,self.start_pos_z]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.plane = p.loadURDF("plane.urdf")
        #self.plane = p.loadURDF((os.path.join(path, "urdf/plane.urdf")))
        self.robot = p.loadURDF(os.path.join(path, "urdf/robot.urdf"),
                                cubeStartPos,
                                cubeStartOrientation, flags=p.URDF_USE_SELF_COLLISION|p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self._observation = self._calculate_observation()
        self._calculate_progress()
        for i in range(8):
            p.enableJointForceTorqueSensor(self.robot,i)
        if self.isDebug:
            _link_name_to_index = {p.getBodyInfo(self.robot)[0].decode('UTF-8'):-1,}
            for _id in range(p.getNumJoints(self.robot)):
                _name = p.getJointInfo(self.robot, _id)[12].decode('UTF-8')
                _link_name_to_index[_name] = _id
                print(f'Link: {_name} {_id}')
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
        for _ in range(int(1/(self.dt*self.freq))):
            self._apply_action(action)
            p.stepSimulation()
        observation = self._calculate_observation()
        reward = self._calculate_reward()
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
                            targetPosition=action[i],
                            maxVelocity=self.maxVelocity)
        

    def _calculate_reward(self):
        alive_coef = 1.0
        feet_collissions_coef = 1.0
        progres_coef = 1.0
        electricity_coef = 1.0
        stall_torque_coef = 1.0
        joints_at_limits_coef = 1.0
        alive = self._calculate_alive_bonus()
        collisions = self._calculate_feet_collisions()
        progress = self._calculate_progress()
        electricity_cost = self._calculate_electricity_cost()
        stall_torque_cost = self._calculate_stall_torque_cost()
        joints_at_limits = self.calculate_joints_at_limits()

        reward = alive_coef*alive + progres_coef*progress + feet_collissions_coef * \
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
        return self._alive < 0


    def _calculate_feet_collisions(self):
        links = {1:"front_left",
                3:"rear_left",
                5:"front_right",
                7:"rear_right"}
        collision_points = 0
        for i in links.keys():
            cp = p.getContactPoints(self.robot,self.robot,i)
            collision_points += len(cp)
            # if self.isDebug:
            #     for point in cp:
            #         print(f'Collision point {links[i]}: {point}')
        if self.isDebug:
            print(f'Number of collisions:{collision_points}')
        if collision_points > 0:
            return -1
        else:
            return +1
           
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
            # if self.isDebug:
            #     for point in cp:
            #         print(f'Contact point {links[i]}: {point}')
        if ground_not_feet_contacts > 0:
            self._alive = -1
        else:
            self._alive = 1
        if self.isDebug:
            print(f'Number of CP:{ground_not_feet_contacts}, Alive: {self._alive}')
        return self._alive

    def _calculate_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        (vx, vy, vz), _ = p.getBaseVelocity(self.robot)
        base_velocity = np.array([vx,vy,vz])
        feet_ground_contacts = []
        links = {1:"front_left",
                3:"rear_left",
                5:"front_right",
                7:"rear_right"}
        for i in links.keys():
            cp = p.getContactPoints(self.robot,self.plane,i)
            feet_ground_contacts.append(len(cp))
        if self.isDebug:
            print(f'Ground contacts: {feet_ground_contacts}')

        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - position[1], self.walk_target_x - position[0]])

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

        if self.isDebug:
            for i in range(8):
                print(f'Joint positions {i}: {self.joint_positions[i]}')
                print(f'Joint velocities {i}: {self.joint_velocities[i]}')
                print(f'Joint torques {i}: {self.joint_torques[i]}')

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
        observation_array = np.clip(np.concatenate((observation_array,self.joint_positions,self.joint_velocities,feet_ground_contacts)),-5,5)
        if self.isDebug:
            print(f'Observation shape: {np.shape(observation_array)}')
        return observation_array
