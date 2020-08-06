from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
import gym
import mm_walker
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import pybullet_envs
tensorboard_log = "rl-quadruped/training/tensorboard"

model_filepath = "rl-quadruped/training/trainedPPO"

env_id = "mm-walker-v0"
env = gym.make(env_id)
env.seed(0)
env = DummyVecEnv([lambda: env]) 
model = PPO2.load(model_filepath, env=env,
                          tensorboard_log=tensorboard_log, verbose=1)
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print(mean_reward, std_reward)