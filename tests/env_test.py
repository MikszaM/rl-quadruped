
import gym
import mm_walker
import time

def loop(env):
    for _ in range(50):
        input()
        sample = env.action_space.sample()
        print(sample)
        env.step(sample)


def main():
    # create the environment
    env = gym.make("mm-walker-v0")
    env.render()
    env.debug()
   # env.log("test2.mp4")
    res = env.reset()
    print(res)
    loop(env)
 

if __name__=="__main__":
    main()