from gym.envs.registration import register

register(
    id='Cartpole-param-v0',
    entry_point='gym_param.envs:CartpoleParamEnv',
)
