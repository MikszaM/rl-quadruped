
import gym
import mm_walker
import pybullet as p
import pybullet_data
import os
import time
from stable_baselines import PPO2
from gym.utils import seeding
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecVideoRecorder

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")



def main():
    #model_filepath = "rl-quadruped/training/save/best_model/best_model"
    #vec_log = "rl-quadruped/training/save/best_model/best_model.pkl"
    model_filepath = "rl-quadruped/training/trainedPPO.zip"
    vec_log = "rl-quadruped/training/trainedPPO.pkl"

    model = PPO2.load(model_filepath, policy=CustomPolicy)
    
    env = gym.make("mm-walker-v0",render=True, logDir="rl-quadruped/training/videos/")
   
    env.seed(0)
    
    env = DummyVecEnv([lambda: env]) 
    env = VecNormalize.load(vec_log, env)
    env.training = False
    env.norm_reward = False

    

    obs = env.reset()
    done = False
    print("Start")
    start = time.time()
    while not done:
        #input()
        elapsed = time.time()-start
        while(elapsed<0.2):
            elapsed = time.time()-start
        print(elapsed)
        start = time.time()

        action, state = model.predict(obs, state=None, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        

if __name__ == "__main__":
    main()