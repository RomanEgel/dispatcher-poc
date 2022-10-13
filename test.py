from gym.wrappers import FlattenObservation
import gym

env = gym.make('dispatcher_env.impl:DispatcherEnv-v0')
wrapped_env = FlattenObservation(env)
print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
