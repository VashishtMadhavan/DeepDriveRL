import gym
import universe 
from universe.wrappers import Monitor
from universe.wrappers import Vectorize, Vision, Logger
from universe.wrappers.experimental import CropObservations, SafeActionSpace

import matplotlib.pyplot as plt
import sys, argparse
import numpy as np
import pygame
import pygame.locals as pl
import h5py

"""
Script to record human gameplay demonstrations for OpenAI Universe

"""

KEYS = [pl.K_SPACE, pl.K_UP, pl.K_RIGHT, pl.K_LEFT, pl.K_DOWN]

def update_keystates(keystates):
    events = pygame.event.get()
    for event in events:
        if hasattr(event, 'key') and event.key == pl.K_ESCAPE:
            exit(0)
        if hasattr(event, 'key') and event.key in KEYS:
            if event.type == pygame.KEYDOWN:
                keystates[event.key] = True
            elif event.type == pygame.KEYUP:
                keystates[event.key] = False


# TODO: make this more gen purpose; only works for dusk drive
def keystates_to_actions(keystates, action_set):
	if keystates[pl.K_UP]:
		return action_set[0]
	elif keystates[pl.K_RIGHT]: 
		return action_set[2]
	elif keystates[pl.K_LEFT]:
		return action_set[1]
	#take random action
	return env.action_space.sample()


def record(args, env):
	keystates = {key: False for key in KEYS}
	observation_n = env.reset()
	episodes = 0
	action_set = env.action_space.actions

	#demonstration recorders
	observations = []
	actions = []
	rewards = []

	while len(observations) < args.max_obs:
		env.render()
		if observation_n[0] is None:
			observation_n, reward_n, done_n, info = env.step([env.action_space.sample() for _ in observation_n])
			continue 

		update_keystates(keystates)
		action = keystates_to_actions(keystates, action_set)
		observation_n, reward_n, done_n, info = env.step([action for _ in observation_n])

		observations.append(observation_n)
		actions.append(action_set.index(action))
		rewards.append(reward_n[0])

		if done_n[0]:
			print "PEACE"
			episodes += 1
			if args.episodes > 0 and episodes >= args.episodes:
				break
			observation_n = env.reset()

	with h5py.File(args.output_file, 'w', libver='latest') as f:
		n = len(observations)
		input_shape = np.array(observations)[0].shape
		task = f.create_dataset('task', data=np.string_(args.task))
		o = f.create_dataset('obs', (n, ) + input_shape, dtype='uint8', compression='lzf', data=np.array(observations))
		a = f.create_dataset('actions', (n, ), dtype='uint8', data=np.array(actions))
		r = f.create_dataset('rewards', (n, ), dtype='uint8', data=np.array(rewards))


def setup(args):
	np.random.seed(args.seed)
	env = gym.make(args.task)

	# Only needed for universe tasks, whole script can be applied to any task
	env = Vision(env)
	env = CropObservations(env)
	env = SafeActionSpace(env)
	env.configure(remotes=1)
	pygame.init()
	return env 

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, default="flashgames.DuskDrive-v0", help="what task you wish to control")
	parser.add_argument('--seed', type=int, default=0, help="seed for random gym starts")
	parser.add_argument('--episodes', type=int, default=1, help="number of demonstration episodes to collect")
	parser.add_argument('--output_file', type=str, default="demonstrations.h5", help="where to save the demonstrations")
	parser.add_argument('--max_obs', type=int, default=5000, help="max number of observations you wish to store")
	return parser.parse_args()

if __name__=="__main__":
	args = parse_args()
	env = setup(args)
	record(args, env)
