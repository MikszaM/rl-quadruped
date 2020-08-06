
import gym
import pybullet_envs
import os 
import pybullet as p
import pybullet_data
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common import make_vec_env
import mm_walker


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

env_id = "mm-walker-v0"
num_cpu = 4 
total_timesteps = 2500

env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                    clip_obs=10.)

model = PPO2('MlpPolicy', env, verbose=1)
print("Learning started")
model.learn(total_timesteps=200)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "tmp/"
model.save(log_dir + "mm")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

# To demonstrate loading
del model, env

# Load the agent
model = PPO2.load(log_dir + "mm")

# Load the saved statistics
env = DummyVecEnv([lambda: gym.make("mm-walker-v0")])
env = VecNormalize.load(stats_path, env)
# do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

env.render()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)