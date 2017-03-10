import gym

env = gym.make('MountainCar-v0')

#print(env.action_space)
#print(env.observation_space.high)
#print(env.observation_space)

'''
episode_length = 0
done = False
env.reset()
while(not done):
	env.render()
	if episode_length == 300:
		print('Done')
		break
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)
	#print('Action', action)
	#print('Observation', observation)
	#print('Reward', reward)
	#print('Info', info)
	#print('')
	episode_length += 1
	print(str(episode_length), done)

print('Episode Length is', str(episode_length))
'''

observation = env.reset()
episode = []
done = False

while(not done):
	env.render()
	episode.append(observation)
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)

print(len(episode))
