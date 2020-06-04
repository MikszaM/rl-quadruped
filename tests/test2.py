import gym
import pybullet_envs


env = gym.make('AntBulletEnv-v0')
env.render()
env.reset()

sam = env.action_space.sample()
print(env.action_space)


for _ in range(100000):
    env.step(env.action_space.sample())
    #env.step([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    pass
env.close()