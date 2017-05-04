import gym
import universe
from universe.wrappers.experimental import SafeActionSpace

env = gym.make('flashgames.DuskDrive-v0')
env = SafeActionSpace(env)
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

while True:
	env.render()
	action_n = [env.action_space.sample() for _ in observation_n]
	observation_n, reward_n, done_n, info = env.step(action_n)
	if done_n[0]:
		break
