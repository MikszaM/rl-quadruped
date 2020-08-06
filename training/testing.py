
import gym
import mm_walker
import pybullet as p
import pybullet_data
import os
from stable_baselines import PPO2
from gym.utils import seeding

def main():
    model_filepath = "rl-quadruped/training/trainedPPO"

    model = PPO2.load(model_filepath)
    env = gym.make("mm-walker-v0")
    env.render()
    env.debug()
    env.log("rl-quadruped/training/trained.mp4")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

if __name__ == "__main__":
    main()