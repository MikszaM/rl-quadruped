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
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 0
        self.walk_target_y = 1e3
        self.isRender = render
        self.logVideo = False
        self.action_space = 0
        self.observation_space = 0
        self._seed()


    def step(self, action):
        print("Step")

    def reset(self):
        if self.isRender:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, -100)
        p.setTimeStep(0.01)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        if self.logVideo and self.isRender:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.logDir)
        cubeStartPos = [0,0,0.315]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.plane = p.loadURDF("plane.urdf")
        path = os.path.abspath(os.path.dirname(__file__))
        self.robot = p.loadURDF(os.path.join(path, "urdf/robot.urdf"),
                           cubeStartPos,
                           cubeStartOrientation)
        self._observation = self._calc_observation()
        return self._observation
        
    def log(self, dir):
        self.logDir = dir
        self.logVideo = True
        

    def render(self, mode='human', close=False):
        self.isRender = True

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _calc_observation(self):
        return 0
