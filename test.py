import gym

env = gym.make('dispatcher_env.impl:DispatcherEnv-v0', render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=42)

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()
    print(observation)

env.close()