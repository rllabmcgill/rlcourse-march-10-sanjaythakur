import gym
from gym import wrappers

env = gym.make('MountainCarModified-v0')
env = wrappers.Monitor(env, './recordings/mountain_car', force=True)

episode = []
for i_episode in range(10):
    observation = env.reset()
    print(len(episode))
    episode = []
    done = False
    while(not done):
        #env.render()
        episode.append(observation)
        if(len(episode) == 300):
            print('Done')
        #    break
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    print(len(episode))
    break
