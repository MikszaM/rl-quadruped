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

from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from gym import spaces
import numpy as np

model_filepath = "rl-quadruped/training/trainedPPO"
vec_log = "rl-quadruped/training/trainedPPO.pkl"
tensorboard_log = "rl-quadruped/training/tensorboard"
save_dir = "rl-quadruped/training/save"

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix=None, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, '{}_{}_steps.pkl'.format(self.name_prefix, self.num_timesteps))
            else:
                path = os.path.join(self.save_path, 'best_model.pkl')
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print("Saving VecNormalize to {}".format(path))
        return True

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
    total_timesteps = 750000
    checkpoint_frequency = 2500
    eval_frequency = 5000
   
    register_policy('CustomPolicy', CustomPolicy)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency, save_path=save_dir, verbose=1)
    global_save_normalization_callback = SaveVecNormalizeCallback(checkpoint_frequency,save_dir,name_prefix="rl_model",verbose=1)
    best_model_save_normalization_callback = SaveVecNormalizeCallback(1,save_dir+"/best_model",verbose=1)
    # Separate evaluation env
    eval_env = gym.make(env_id)
    eval_env = DummyVecEnv([lambda: eval_env]) 
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir+"/best_model",  callback_on_new_best=best_model_save_normalization_callback,
                                log_path=save_dir+"/results", eval_freq=eval_frequency, render=False, deterministic=True, verbose=1, n_eval_episodes=1)
    # Tensorboard callback
    tb_callback = TensorboardCallback()
    # Create the callback list
    callback = CallbackList([checkpoint_callback,global_save_normalization_callback, eval_callback, tb_callback])
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    if first_run:
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
    else:
        env = VecNormalize.load(vec_log, env)
    if first_run:
        model = PPO2(CustomPolicy, env, verbose=1,
                     tensorboard_log=tensorboard_log)
        print("Learning is about to start")
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        model = PPO2.load(model_filepath, env=env,
                          tensorboard_log=tensorboard_log, verbose=1)
        model.learn(total_timesteps=total_timesteps,
                    callback=callback, reset_num_timesteps=False)

    model.save(model_filepath)
    env.save(vec_log)
    print(f"Saving to : {model_filepath}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main(True)

# 4 hours training
