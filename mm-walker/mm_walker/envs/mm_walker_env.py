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

        self.action_space = spaces.Box(
            low=np.array([-0.2,-2.4,-2.4,-0.2,-2.2,-1.9,-2.2,-1.2]),
            high=np.array([2.4,0.2,0.2,2.4,1.9,2.2,1.9,2.2]), 
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(25,), dtype=np.float32)
        self._observation = []

        self.dt = 1./2400.
        self.freq = 10
        self.maxVelocity = 5.1
        self.potential = 0
        self._alive = 1
        self.isDebug = False
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 0
        self.walk_target_y = 1e3
        self.isRender = render
        self.logVideo = False
        self._seed()


    def step(self, action):
        for _ in range(int(1/(self.dt*self.freq))):
            self._apply_action(action)
            p.stepSimulation()
        observation = self._calculate_observation()
        done = self._calculate_done()
        reward = self._calculate_reward()
        
        if self.isDebug:
            print(f"Observation: {observation}")
            print(f"Done: {done}")
            print(f"Reward: {reward}")
        return observation, reward, done, {}

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
        cubeStartPos = [0,0,0.315]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.plane = p.loadURDF("plane.urdf")
        #self.plane = p.loadURDF((os.path.join(path, "urdf/plane.urdf")))
        self.robot = p.loadURDF(os.path.join(path, "urdf/robot.urdf"),
                           cubeStartPos,
                           cubeStartOrientation)
        self._observation = self._calculate_observation()
        for i in range(8):
            p.enableJointForceTorqueSensor(self.robot,i)
        return self._observation
        
    def log(self, dir):
        self.logDir = dir
        self.logVideo = True
        

    def render(self, mode='human', close=False):
        self.isRender = True

    def debug(self):
        self.isDebug = True


    def _apply_action(self,action):
        for i in range(8):
            p.setJointMotorControl2(self.robot, i,
                            p.POSITION_CONTROL,
                            targetPosition=action[i],
                            maxVelocity=self.maxVelocity)
        


    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def _calculate_reward(self):
        potential_old = self.potential
        self.potential = self.walk_target_y/(self.dt*self.freq)
        progress = float(self.potential - potential_old)
        return progress

    def _calculate_done(self):
        return self._alive < 0


    def _calculate_alive_bonus(self):
        self._alive = 1

    def _calculate_observation(self):
        basePosAndOri = np.array(p.getBasePositionAndOrientation(self.robot))
        jointStates = p.getJointStates(self.robot, range(8))
        baseVelocity = p.getBaseVelocity(self.robot)
        return basePosAndOri
