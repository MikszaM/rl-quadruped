import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import time
import numpy as np


class MMWalkerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    pass

  def step(self, action):
    pass

  def reset(self):
    pass

  def _render(self, mode='human', close=False):
    pass

  def _seed(self):
    pass
