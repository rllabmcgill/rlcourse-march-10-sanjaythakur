import gym
from gym import wrappers
env = gym.make('MountainCar-v0')
#env = wrappers.Monitor(env, '/home/sanjay/Course Tasks/Reinforcement Learning/Task 4/recordings/mountain_car')
episode = []
for i_episode in range(20000):
    observation = env.reset()
    if len(episode) < 200:
        print('Yes', str(i_episode))
    print(len(episode))
    episode = []
    done = False
    while(not done):
        #env.render()
        episode.append(observation)
        #if(len(episode) == 300):
        #    print('Done')
        #    break
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
