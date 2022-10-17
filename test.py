import gym
import dispatcher_env.impl

env = gym.make('DispatcherEnv-v0', render_mode="human")
env.action_space.seed(42)
observation = env.reset(seed=42)

for _ in range(100):
    observation, reward, terminated, truncated = env.step(env.action_space.sample())

    if terminated or truncated:
        observation = env.reset()
    print(observation)

env.close()