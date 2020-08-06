import gym
import mm_walker
import pybullet as p
import pybullet_data
import pybullet_envs
import os
import time
from gym.utils import seeding
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from gym import spaces
import numpy as np

model_filepath = "rl-quadruped/training/trainedPPO"
tensorboard_log = "rl-quadruped/training/tensorboard"
save_dir = "rl-quadruped/training/save"

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.mean(self.model.episode_reward)
        summary = tf.Summary(value=[tf.Summary.Value(
            tag='episode_reward', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def main(first_run):
    start_time = time.time()
    env_id = "mm-walker-v0"
    num_cpu = 4
    total_timesteps = 2000

    checkpoint_callback = CheckpointCallback(
        save_freq=10, save_path=save_dir, verbose=1)
    # Separate evaluation env
    #eval_env = gym.make(env_id)
    #eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir+"/best_model",
     #                            log_path=save_dir+"/results", eval_freq=50,render=False,deterministic=True, verbose=1, n_eval_episodes=1)
    # Tensorboard callback
    tb_callback = TensorboardCallback()
    # Create the callback list
    #callback = CallbackList([checkpoint_callback, eval_callback, tb_callback])
    callback = CallbackList([checkpoint_callback, tb_callback])

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    if first_run:
        model = PPO2(MlpPolicy, env, verbose=1,
                     tensorboard_log=tensorboard_log)
        print("Learning is about to start")
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        model = PPO2.load(model_filepath, env=env,
                          tensorboard_log=tensorboard_log, verbose=1)
        model.learn(total_timesteps=total_timesteps,
                    callback=callback, reset_num_timesteps=False)

    model.save(model_filepath)
    print(f"Saving to : {model_filepath}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main(True)

# 4 hours training
