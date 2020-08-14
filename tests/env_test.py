
import gym
import mm_walker
import time

def loop(env):
    for _ in range(50):
        input()
        if _%15==0:
            start = time.time()
            res = env.reset()
            elapsed_time = time.time() - start
            print(elapsed_time)
            print(res)
        else:
            sample = env.action_space.sample()
            print(sample)
            env.step(sample)


def main():
    # create the environment
    env = gym.make("mm-walker-v0",render=True)
    env.debug()
   # env.log("test2.mp4")
    start = time.time()
    res = env.reset()
    elapsed_time = time.time() - start
    print(elapsed_time)
    print(res)
    loop(env)
 

if __name__=="__main__":
    main()