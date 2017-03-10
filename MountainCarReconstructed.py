import gym
from gym import wrappers
import numpy as np
import random

env = gym.make('MountainCarModified-v0')
env = wrappers.Monitor(env, './recordings/mountain_car', force=True)


