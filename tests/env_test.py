
import gym
import mm_walker
import time

def loop(env):
    for _ in range(100000):
        sample = env.action_space.sample()
        env.step(sample)


def main():
    # create the environment
    env = gym.make("mm-walker-v0")
    env.render()
    # env.log("test2.mp4")
    env.reset()
    env.debug()
    loop(env)
 

if __name__=="__main__":
    main()