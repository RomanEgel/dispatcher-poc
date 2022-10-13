from gym.envs.registration import register

register(
    id='DispatcherEnv-v0',
    entry_point='dispatcher_env.impl.envs:DispatcherEnv',
    max_episode_steps=300
)