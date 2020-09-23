from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
import gym
import mm_walker
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import pybullet_envs

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

tensorboard_log = "rl-quadruped/training/tensorboard"

model_filepath = "rl-quadruped/training/trainedPPO.zip"
veclog = "rl-quadruped/training/trainedPPO.pkl"
env_id = "mm-walker-v0"
env = gym.make(env_id,render=True)
env.seed(0)
env = DummyVecEnv([lambda: env]) 
#env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
env = VecNormalize.load(veclog, env)
env.training = False
env.norm_reward = False
model = PPO2.load(model_filepath, env=env,
                          tensorboard_log=tensorboard_log, verbose=1, policy=CustomPolicy)
# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env,  deterministic=True, n_eval_episodes=1)

print(mean_reward, std_reward)