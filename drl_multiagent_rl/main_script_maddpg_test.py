

import os
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np
from collections import  Counter

from  framework.ddpg_agent import Agent
from  framework.ddpg_model import *

import pandas as pd







if __name__=='__main__':


    version = 1 # version(int): version for the model
    env = UnityEnvironment(file_name=os.path.join('env','Reacher.app'))

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

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

    scores = np.zeros(num_agents)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random_seed = 0
    Agent.actor_local = Actor(state_size, action_size, random_seed).to(device)
    Agent.actor_local.load_state_dict(torch.load(os.path.join('model', 'checkpoint_actor_' + str(version) + '.pth')))

    agents = []

    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, random_seed=0))

    while True:
        actions = np.array([agents[i].agents_act(states[i]) for i in range(num_agents)])

        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done

        states = next_states
        scores += rewards

        print('\rScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
              .format(np.mean(scores), np.min(scores), np.max(scores)), end="")

        if np.any(dones):
            break

    print("\nScores: {}".format(scores))

    env.close()



