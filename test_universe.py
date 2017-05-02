import gym
import universe 
import sys
from universe.wrappers import Monitor
from universe.wrappers import Vectorize, Vision, Logger
from universe.wrappers.experimental import CropObservations, SafeActionSpace
import matplotlib.pyplot as plt
import numpy as np
import random

env = gym.make('flashgames.DuskDrive-v0')
#env = Logger(env)
#env = Monitor(env, "temp_monitor", force=True)
env = Vision(env)
env = CropObservations(env)
env = SafeActionSpace(env)

env.configure(remotes=1) 
#env.configure(remotes='vnc://localhost:5900+15900')
# automatically creates a local docker container
#actions = env.action_space.actions
#observation_n = env.reset()
#ounter = 0
#f = open("train.log",'w')

while counter < 3000:
#  action_n = [random.sample(actions,1)[0] for ob in observation_n]
#  observation_n, reward_n, done_n, info = env.step(action_n)
#  f.write("Counter: %s\n" % (str(counter)))
#  f.write("Reward: %s\n" %(str(reward_n[0])))
  counter+=1
  env.render()
