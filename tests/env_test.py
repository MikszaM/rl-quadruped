
import gym
import mm_walker
import time

def main():
    # create the environment
    env = gym.make("mm-walker-v0")
    env.render()
    env.log("test2.mp4")
    env.reset()
    time.sleep(5)
 

if __name__=="__main__":
    main()