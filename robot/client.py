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

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")



def main():
    #model_filepath = "rl-quadruped/training/save/best_model/best_model"
    #vec_log = "rl-quadruped/training/save/best_model/best_model.pkl"
    model_filepath = "rl-quadruped/training/trainedPPO"
    vec_log = "rl-quadruped/training/trainedPPO.pkl"

    model = PPO2.load(model_filepath, policy=CustomPolicy)
    
    env = gym.make("mm-walker-v0",render=True)
    env.render()
    env.seed(0)
    env = DummyVecEnv([lambda: env]) 
    env = VecNormalize.load(vec_log, env)
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    done = False
    HOST = '192.168.0.220'
    PORT = 5005


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Start")
        #input()
        start = time.time()
        while not done:
            action, state = model.predict(obs, state=None, deterministic=True)
            action_string = ""
            for a in action[0]:
                action_string += f'{a:.2f},'
            action_string = action_string[:-1]

            elapsed = time.time()-start
            while(elapsed<0.19):
                elapsed = time.time()-start
            

            s.sendall(action_string.encode())
            print(elapsed)
            start = time.time()
            #print(action_string)
            obs, reward, done, info = env.step(action)
            
        s.sendall("0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00".encode())
        

if __name__ == "__main__":
    main()
