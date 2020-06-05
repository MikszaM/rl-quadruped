from gym.envs.registration import register
 
register(
    id='mm-walker-v0',
    entry_point='mm_walker.envs:MMWalkerEnv',
)