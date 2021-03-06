import gym
from gym import wrappers
import numpy as np
import random

#All_constants

#Make sure you change this variable whenever you change the number of features space components
features_per_action = 7

class Agent():
	def __init__(self, env, step_size = 0.05, gamma = 0.95, horizon = 30):
		self.step_size = step_size
		self.env = env
		self.gamma = gamma
		self.weights = ((2 * np.random.ranf([( features_per_action * self.env.action_space.n ) + 1])) - 1)/2.0
		self.EPSILON = 1
		self.HORIZON = horizon

	def doFunctionApproximationFromStateActionPair(self, episode_step):
		feature_vector  = self.getFeatureVector(episode_step)
		return np.dot(feature_vector, self.weights)
		
	def updateFunctionApproximator(self, episode):
		for episode_step_iterator in range(len(episode)):
			#print(str(episode_step_iterator))
			episode_step = episode[episode_step_iterator]
			discounted_return = self.getDiscountedReturn(episode[episode_step_iterator:])
			current_estimated_value = self.doFunctionApproximationFromStateActionPair(episode_step)
			delta_weights = np.multiply(np.multiply(self.step_size, self.getFeatureVector(episode_step)), (discounted_return - current_estimated_value))
			self.weights = self.weights + delta_weights
		print("Updated")

	def getDiscountedReturn(self, episode):
		discounted_return = 0.0
		effective_discounting = 1.0
		horizon_iterator = 0
		for episode_step in episode:
			if horizon_iterator > self.HORIZON:
				break
			discounted_return = discounted_return + (effective_discounting * episode_step[-1])
			effective_discounting *= self.gamma
			horizon_iterator += 1
		return discounted_return

	def start(self):
		observation = self.env.reset()
		episode_counter = 0
		while(True):
			print('Episode number', str(episode_counter))
			random_throw = random.uniform(0, 1)
			if random_throw < self.EPSILON:
				episode = self.drawEpisodeStochastically()
			else:
				episode = self.drawEpisodeGreedily()
			self.updateFunctionApproximator(episode)
			episode_counter += 1
			self.EPSILON = max(0.25, (self.EPSILON * 0.95))

	def drawEpisodeStochastically(self):
		episode = []
		print('Stochastic')
		observation = self.env.reset()
		done = False
		while(not done):
			#self.env.render()
				
			episode_step = [observation]

			action = self.env.action_space.sample()
			episode_step.append(action)

			observation, reward, done, info = self.env.step(action)
			episode_step.append(reward)
			episode.append(episode_step)

			#print('Step taken')

		return episode

	def drawEpisodeGreedily(self):
		episode = []
		print('Greedy')
		observation = self.env.reset()
		done = False
		while(not done):
			#self.env.render()
			episode_step = [observation]

			best_action = self.env.action_space.sample()
			best_value = self.doFunctionApproximationFromStateActionPair([observation, best_action])
			for action_iterator in range(self.env.action_space.n):
				value = self.doFunctionApproximationFromStateActionPair([observation, action_iterator])
				if value > best_value:
					best_action = action_iterator
					best_value = value

			episode_step.append(best_action)

			observation, reward, done, info = self.env.step(best_action)
			episode_step.append(reward)

			episode.append(episode_step)

			#print('Step taken')			

		return episode

	def getFeatureVector(self, episode_step):
		observation = episode_step[0]
		action = episode_step[1]
		feature_vector = [1]
		for action_iterator in range(self.env.action_space.n):
			if action_iterator == action:
				feature_vector.append(observation[0])
				feature_vector.append(observation[1])
				feature_vector.append(observation[0]*observation[1])
				feature_vector.append(observation[0]*observation[0])
				feature_vector.append(observation[1]*observation[1])
				feature_vector.append(observation[0]*observation[1]*observation[1])
				feature_vector.append(observation[0]*observation[0]*observation[1])
			else:
				for feature_iterator in range(features_per_action):
					feature_vector.append(0)
		return feature_vector


env = gym.make('MountainCarModified-v0')
env = wrappers.Monitor(env, './recordings/mountain_car', force=True)
agent = Agent(env, step_size= 0.05, gamma = 1)
agent.start()
