import gym
import mm_walker

def loop(env):
    for _ in range(50):
        sample = env.action_space.sample()
        print(sample)
        env.step(sample)


def main():
    env = gym.make("mm-walker-v0",render=True, logDir="rl-quadruped/training/videos/Random/")
    res = env.reset()
    print(res)
    loop(env)
 

if __name__=="__main__":
    main()