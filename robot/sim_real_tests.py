import socket
import time
import gym
import mm_walker
import pybullet as p
import pybullet_data
import os
import time
from stable_baselines import PPO2
from gym.utils import seeding
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

def main():
   
    env = gym.make("mm-walker-v0",render=True)
    env.render()
    env.seed(0)
    env.log("rl-quadruped/training/trained.mp4")
    env = DummyVecEnv([lambda: env]) 
    vec_log = "rl-quadruped/training/trainedPPO.pkl"
    env = VecNormalize.load(vec_log, env)
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    done = False
    HOST = '192.168.0.220'
    PORT = 5005
    start = time.time()

    test_actions = []
    test_actions.append([[1,0.5,1,0.5,1,0.5,1,0.5]])
    test_actions.append([[0,0.5,0,0.5,0,0.5,0,0.5]])
    test_actions.append([[-0.5,0.5,-0.5,0.5,-0.5,0.1,-0.5,0.1]])
    test_actions.append([[1,0.5,1,0.5,1,0.5,1,0.5]])
    test_actions.append([[0,0.5,0,0.5,0,0.5,0,0.5]])
    test_actions.append([[-0.5,0.5,-0.5,0.5,-0.5,0.1,-0.5,0.1]])
    test_actions.append([[1,0.5,1,0.5,1,0.5,1,0.5]])
    test_actions.append([[0,0.5,0,0.5,0,0.5,0,0.5]])
    test_actions.append([[-0.5,0.5,-0.5,0.5,-0.5,0.1,-0.5,0.1]])
    test_actions.append([[1,0.5,1,0.5,1,0.5,1,0.5]])
    test_actions.append([[0,0.5,0,0.5,0,0.5,0,0.5]])
    test_actions.append([[-0.5,0.5,-0.5,0.5,-0.5,0.1,-0.5,0.1]])
    test_actions.append([[1,0.5,1,0.5,1,0.5,1,0.5]])
    test_actions.append([[0,0.5,0,0.5,0,0.5,0,0.5]])
    test_actions.append([[-0.5,0.5,-0.5,0.5,-0.5,0.1,-0.5,0.1]])
    test_actions.append([[0,0.5,0,0.5,0,0.5,0,0.5]])
    i=0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Start")
        input()
        while i<len(test_actions):
            action = test_actions[i]
            action_string = ""
            for a in action[0]:
                action_string += f'{a:.2f},'
            action_string = action_string[:-1]
            s.sendall(action_string.encode())
            print(action_string)
            obs, reward, done, info = env.step(action)
            elapsed = time.time() - start
            start = time.time()
            print(elapsed)
            print(reward)
            i+=1
    print("koniec")
    input()
   
        

if __name__ == "__main__":
    main()
