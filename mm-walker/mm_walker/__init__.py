from gym.envs.registration import register
 
register(
    id='mmwalker-v0',
    entry_point='mm_walker.envs:MMWalkerEnv',
)