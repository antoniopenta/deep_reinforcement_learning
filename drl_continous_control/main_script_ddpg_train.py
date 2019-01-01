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
from  framework.ddpg_agent import Agent

import pandas as pd






if __name__=='__main__':

    #seed (int) : seed used to inizialize the network
    #version (int): version number to save model check point and files
    #report_value (int): report value used to average the last values

    seed = 12
    version = 1
    report_value = 10
    file_scores = os.path.join('data','scores_'+str(version)+'.txt')


    # n_episodes (int): maximum number of training episodes
    # max_t (int): maximum number of timesteps per episode
    # eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    # eps_end (float): minimum value of epsilon
    # eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    # max_score (float): that target score that we would like to reach, the benchmark is 15 in 1700 episode
    #drop_p(float): Dropout Probability

    n_episodes = 110
    max_t = 500
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 0.995
    max_score = 30


    env = UnityEnvironment(file_name=os.path.join('env','Reacher.app'))
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]



    ####


    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    ##



    #instanziate the agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)

    # DDPG training

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=report_value)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        # reset the env
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
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

        if i_episode % report_value == 0:
            print('\n')
            print('-' * 100)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\n')
            print('-' * 100)

        if np.mean(scores_window) >= max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - report_value,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), os.path.join('model','checkpoint_actor_'+str(version)+'.pth'))
            torch.save(agent.critic_local.state_dict(),
                       os.path.join('model', 'checkpoint_critic_' + str(version) + '.pth'))
            break

    if np.mean(scores_window) < max_score:
        print('\nEnvironment not fully solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - report_value,
                                                                                     np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(), os.path.join('model', 'checkpoint_actor_' + str(version) + '.pth'))
        torch.save(agent.actor_local.state_dict(), os.path.join('model', 'checkpoint_critic_' + str(version) + '.pth'))

    with open(file_scores, 'w') as fscores:
        fscores.write('\n'.join([str(item) for item in scores]))

    env.close()


