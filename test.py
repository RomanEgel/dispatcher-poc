from gym.wrappers import FlattenObservation
import gym
import gym_examples.gym_examples

env = gym.make('GridWorld-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
