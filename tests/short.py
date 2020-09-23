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
        ...


    def reset(self):
        ...
        return self._observation


    def render(self, mode='human', close=False):
        ...


    def step(self, action):  
        self._apply_action(action)
        observation = self._calculate_observation()
        reward = self._calculate_reward()
        done = self._calculate_done()
        return observation, reward, done, {}


    def _apply_action(self,action):
        ...


    def _calculate_observation(self, calculate_feet_contacts=True):
        ...
        return observation_array


    def _calculate_reward(self):
        ...
        return reward


    def _calculate_done(self):
        ...
        return done