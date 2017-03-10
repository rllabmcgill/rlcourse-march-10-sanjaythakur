import gym
import numpy as np, pandas as pd

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")
env.reset()

def generate_session(agent, n_actions, t_max=1000):
    
    states,actions = [],[]
    total_reward = 0
    
    s = env.reset()
    
    for t in range(t_max):
        
        #predict array of action probabilities
        probs = agent.predict_proba([s])[0] 
        
        a = np.random.choice(range(n_actions), size=1, p=probs)[0]
        
        new_s,r,done,info = env.step(a)
        
        #record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward+=r
        
        s = new_s
        if done:
            break
    return states,actions,total_reward

agent = MLPClassifier(hidden_layer_sizes=(42,42),
                      activation='tanh',
                      warm_start=True, #keep progress between .fit(...) calls
                      max_iter=2 )
n_ticks = 180
n_samples=100
percentile=70
smoothing=0.01

n_actions = env.action_space.n
mean_rewards = []
thresholds = []
agent.fit([env.reset()]*n_actions,range(n_actions));

history_sessions = []

for i in range(n_ticks):
    #generate new sessions
    sessions = [generate_session(agent, n_actions, 15000) for _ in range(n_samples)]
    
    batch_rewards = np.array([sample[2] for sample in sessions]) 
    mean_score = np.mean(batch_rewards)
    threshold = np.percentile(batch_rewards, percentile)
    
    history_sessions = filter(lambda x: x[2] > threshold, history_sessions)
    if len(history_sessions) == 0:
        history_sessions = filter(lambda x: x[2] >= threshold, history_sessions)
    elite_sessions = filter(lambda x: x[2] > threshold, sessions)
    if len(elite_sessions) == 0:
        elite_sessions = filter(lambda x: x[2] >= threshold, sessions)
    history_sessions.extend(elite_sessions)
    if len(history_sessions) > n_samples:
        history_sessions = history_sessions[-n_samples:]
    print("Take ", len(history_sessions) - len(elite_sessions), "old sessions")

    elite_states,elite_actions,batch_rewards = map(np.array,zip(*history_sessions))
    #batch_states: a list of lists of states in each session
    #batch_actions: a list of lists of actions in each session
    #batch_rewards: a list of floats - total rewards at each session

    elite_states, elite_actions = map(np.concatenate,[elite_states,elite_actions])
    #elite_states: a list of states from top games
    #elite_actions: a list of actions from top games
    
    agent.fit(elite_states, elite_actions)
    mean_rewards.append(np.mean(batch_rewards))
    thresholds.append(np.mean(threshold))
    print("%d\tmean reward = %.5f\tthreshold = %.1f"%(i, mean_score,threshold))

env=gym.make("MountainCar-v0");env.reset();
import gym.wrappers
env = gym.wrappers.Monitor(env,directory="videos",force=True)
sessions = [generate_session(agent, n_actions, 15000) for _ in range(100)]
env.close()
#unwrap 
env = env.env.env
