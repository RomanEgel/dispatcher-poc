import gym
import dispatcher_env.impl
from stable_baselines3.common.env_checker import check_env

env = gym.make('DispatcherEnv-v0', render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=42)

check_env(env)

for _ in range(100):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation = env.reset()
    print(observation)

env.close()