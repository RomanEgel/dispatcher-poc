import gym

env = gym.make('dispatcher_env.impl:DispatcherEnv-v0', render_mode="human", new_step_api=True)
env.action_space.seed(42)
observation = env.reset(seed=42)

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation = env.reset()
    print(observation)

env.close()