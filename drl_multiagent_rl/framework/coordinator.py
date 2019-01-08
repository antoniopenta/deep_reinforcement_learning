
import numpy as np
import random
import copy
from collections import namedtuple, deque

from framework.networks import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss
from framework.agent import *
from utils.buffer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPGCoordinator():

    def __init__(self, agent_list, action_size, config):


        self.maddpg_agents = agent_list

        self.config = config

        self.buffer = ReplayBuffer(action_size, self.config.maddpa_buffer_size, self.config.maddpa_batch_size, self.config.maddpa_buffer_random_seed)

        self.num_agents = len(self.maddpg_agents)






    def agents_act(self, states, exploration):
        """get actions from all agents"""
        actions = []
        for agent,state in zip(self.maddpg_agents, states):
            actions.append(agent.local_act(to_tensor(state), exploration=exploration).data.numpy())
        return actions


    def remember(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.add(state, action, reward, next_state, done)



    def reset_noise(self):
        for agent in self.maddpg_agents:
            agent.reset_noise()


    def step(self,t_step):

        if t_step % self.config.ddpg_update_every==0:
            if len(self.buffer) < self.config.maddpa_batch_size:
                return

            for _ in range(0, self.config.ddpg_updates_per_step):
                for agent in self.maddpg_agents:
                    experience = self.buffer.sample()
                    agent.learn(experience,self.config.maddpa_gamma)







