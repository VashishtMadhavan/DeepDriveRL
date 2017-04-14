import gym
import universe 
import sys
from universe.wrappers import Vectorize, Vision, Logger
from universe.wrappers.experimental import CropObservations, SafeActionSpace
import numpy as np
import random

env = gym.make('flashgames.DuskDrive-v0')
env = Vision(env)
env = CropObservations(env)
env = SafeActionSpace(env)
env = Logger(env)
env.configure(remotes=1) 

# automatically creates a local docker container
actions = env.action_space[0] + env.action_space[1] + env.action_space[2]
observation_n = env.reset()
counter = 0

while counter < 3000:
  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
  observation_n, reward_n, done_n, info = env.step(action_n)
  counter+=1
  env.render()

