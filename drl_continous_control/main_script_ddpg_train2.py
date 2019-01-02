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
    # max_score (float): that target score that we would like to reach, the benchmark is 15 in 1700 episode
    # n_episodes (int): maximum number of training episodes
    # noise_selection (boolean): if we would like to add noise to the actions (noise if True)
    #Linux (boolean): boolan value used to run on AWS (aws if Linux = True)
    # logs 0, action, 1 action+noise, 2 rewards


    seed = 15
    version = 7
    report_value = 10
    n_episodes = 300
    max_score = 30.0

    noise_selection = True
    Linux =  True

    logs=[[],[],[]]


    file_scores = os.path.join('data','scores_'+str(version)+'.txt')
    file_actions = os.path.join('data', 'actions_' + str(version) + '.txt')
    file_actions_plus_noise = os.path.join('data', 'actions_noise_' + str(version) + '.txt')
    file_rewards = os.path.join('data', 'rewards_' + str(version) + '.txt')

    if Linux:
        env = UnityEnvironment(file_name=os.path.join(os.getcwd(),'env', 'Reacher_Linux_NoVis','Reacher.x86_64'))
    else:
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
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed,logs=logs)


    avg_scores = []
    scores_deque = deque(maxlen=report_value)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        agent.reset()
        scores = np.zeros(num_agents)
        while True:
            states = env_info.vector_observations   # get the current state
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            rewards = env_info.rewards  # get reward (for each agent)
            scores += rewards
            next_states = env_info.vector_observations  # get next state (for each agent)
            dones = env_info.local_done  # see if episode finished
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            if any(env_info.local_done):
                break

        scores_deque.append(np.mean(scores))
        avg_scores.append(np.mean(scores))

        if i_episode % report_value == 0:
            print('\n')
            print('-' * 100)
            print('\rEpisode {}\tAverage Score: {:.2f} Max Score: {:.2f} , Min Score: {:.2f}'.format(i_episode, np.mean(scores),np.max(scores),np.min(scores)))
            print('\n')
            print('-' * 100)


        if np.mean(scores_deque) >= max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - report_value,
                                                                                         np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(),
                       os.path.join('model', 'checkpoint_actor_' + str(version) + '.pth'))
            torch.save(agent.critic_local.state_dict(),
                       os.path.join('model', 'checkpoint_critic_' + str(version) + '.pth'))
            break

    if np.mean(scores_deque) < max_score:
        print('\nEnvironment not fully solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - report_value,
                np.mean(scores_deque)))
        torch.save(agent.actor_local.state_dict(),
                       os.path.join('model', 'checkpoint_actor_' + str(version) + '.pth'))
        torch.save(agent.actor_local.state_dict(),
                       os.path.join('model', 'checkpoint_critic_' + str(version) + '.pth'))

    with open(file_scores, 'w') as fscores:
        fscores.write('\n'.join([str(item) for item in avg_scores]))

    if logs is not None:
        np.savetxt(file_actions, np.vstack(logs[0]), delimiter=',', fmt='%.4f')
        np.savetxt(file_actions_plus_noise, np.vstack(logs[1]), delimiter=',', fmt='%.4f')
        np.savetxt(file_rewards, np.vstack(logs[2]), delimiter=',', fmt='%.4f')

    env.close()


