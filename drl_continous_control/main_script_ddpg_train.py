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



    version = 1 #version (int): version number to save model check point and files
    report_value = 10
    n_episodes = 300 # n_episodes (int): maximum number of training episodes
    max_score = 30.0  # max_score (float): that target score that we would like to reach, the benchmark is 15 in 1700 episode
    score_window_size = 100 #score_window_size(int): buffer size for the score

    max_t = 1000  # max_t (int) max length of the time stemps
    time_stamp_report = 20  #time_stamp_report(int): value used to report each time stamps

    noise_selection = True # noise_selection (boolean): if we would like to add noise to the actions (noise if True)
    Linux =  False  #Linux (boolean): boolan value used to run on AWS (aws if Linux = True)

    file_scores = os.path.join('data','scores_'+str(version)+'.txt')

    if Linux:
        env = UnityEnvironment(file_name=os.path.join(os.getcwd(),'env', 'Reacher_Linux_NoVis','Reacher.x86_64'))
    else:
        env = UnityEnvironment(file_name=os.path.join('env','Reacher.app'))

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

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

    scores_window = deque(maxlen=score_window_size)
    scores = np.zeros(num_agents)
    scores_episode = []

    agents = []

    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, random_seed=0))

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        for agent in agents:
            agent.reset()

        scores = np.zeros(num_agents)

        for t in range(max_t):
            actions = np.array([agents[i].act(states[i],add_noise=noise_selection) for i in range(num_agents)])
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done
            for i in range(num_agents):
                agents[i].step(t, states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states
            scores += rewards
            if t % time_stamp_report:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(t, np.mean(scores), np.min(scores), np.max(scores)), end="")
            if np.any(dones):
                break
        score = np.mean(scores)
        scores_window.append(score)  # save most recent score
        scores_episode.append(score)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)),
              end="\n")
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - score_window_size,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(),
                       os.path.join('model', 'checkpoint_actor_' + str(version) + '.pth'))
            torch.save(agent.critic_local.state_dict(),
                       os.path.join('model', 'checkpoint_critic_' + str(version) + '.pth'))

            break

    if np.mean(scores_window) < max_score:
        print('\nEnvironment not fully solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - report_value,
                np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(),
                       os.path.join('model', 'checkpoint_actor_' + str(version) + '.pth'))
        torch.save(agent.actor_local.state_dict(),
                       os.path.join('model', 'checkpoint_critic_' + str(version) + '.pth'))

    with open(file_scores, 'w') as fscores:
            fscores.write('\n'.join([str(item) for item in scores_episode]))


    env.close()


