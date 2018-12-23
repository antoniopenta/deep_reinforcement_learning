import os
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np
from collections import  Counter

from framework.model import *
from framework.dqn_agent import *

import pandas as pd






if __name__=='__main__':

    #seed (int) : seed used to inizialize the network
    #version (int): version number to save model check point and files
    #report_value (int): report value used to average the last values

    seed = 12
    version = 3
    report_value = 100
    actions_name = ['move forward', 'move backward', 'turn left', 'turn right']
    file_scores = os.path.join('data','scores_'+str(version)+'.txt')
    file_report = os.path.join('data', 'dataframe_' + str(version) + '.csv')

    array_to_report_full = np.zeros((1,4+3)) # array used to store the distribution of action (4) + average eps + number of episodes + average score


    # n_episodes (int): maximum number of training episodes
    # max_t (int): maximum number of timesteps per episode
    # eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    # eps_end (float): minimum value of epsilon
    # eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    # max_score (float): that target score that we would like to reach, the benchmark is 15 in 1700 episode
    #drop_p(float): Dropout Probability

    n_episodes = 2000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 0.995
    max_score = 20
    drop_p = 0.5

    env = UnityEnvironment(file_name=os.path.join('env','Banana.app'))
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    #reset the env to get the number of states
    env_info = env.reset(train_mode=True)[brain_name]
    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    #instanziate the agent
    agent = Agent(state_size=state_size, action_size=action_size,drop_p=drop_p, seed=seed)

    # Deep Q - Learning training

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=report_value)  # last 100 scores
    eps = eps_start  # initialize epsilon

    action_values = deque(maxlen=report_value)
    eps_values = deque(maxlen=report_value)

    for i_episode in range(1, n_episodes + 1):
        # reset the env
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            action_values.append(action)
            eps_values.append(eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % report_value == 0:
            print('\n')
            print('-'*100)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rEpisode {}\tAverage Eps: {:.2f}'.format(i_episode, np.mean(eps_values)))

            actions_counter = Counter(action_values)
            actions_array = np.zeros((1,action_size))
            for item in actions_counter:
                actions_array[0,item] = actions_counter[item] / len(action_values)
                print('\rEpisode {}\tAction Distribution {} : {:.2f}'.format(i_episode, actions_name[item],
                                                                             actions_array[0,item]))
            average_eps = np.mean(eps_values)
            average_score = np.mean(scores_window)
            array_to_report = np.array([[i_episode,average_eps,average_score]])
            array_to_report = np.hstack([array_to_report,actions_array])
            array_to_report_full = np.vstack([array_to_report_full,array_to_report])
            print('-'*100)

        if np.mean(scores_window) >= max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - report_value,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), os.path.join('model','checkpoint_'+str(version)+'.pth'))
            break

    if np.mean(scores_window) < max_score:
        print('\nEnvironment not fully solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - report_value,
                                                                                     np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), os.path.join('model', 'checkpoint_' + str(version) + '.pth'))

    with open(file_scores, 'w') as fscores:
        fscores.write('\n'.join([str(item) for item in scores]))

    dataframe = pd.DataFrame(array_to_report_full,columns=['episode_num','avg_eps','average_score','move forward', 'move backward', 'turn left', 'turn right'])
    dataframe.to_csv(file_report,index=None)
    env.close()


