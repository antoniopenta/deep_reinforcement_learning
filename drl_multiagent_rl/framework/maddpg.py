
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


class MADDPGLearner:

    def __init__(self,agent_list,config):


        self.maddpg_agents = agent_list

        self.num_agents = len(self.maddpg_agents)

        self.buffer = ReplayBuffer(config.maddpa_buffer_size,config.maddpa_batch_size,self.num_agents,config.buffer_seed)

        self.config = config


    def agents_act(self, states):
        """get actions from all agents"""
        actions = []
        for agent,state in zip(self.maddpg_agents,states):
            actions.append(agent.step(state))
        return actions

    def agents_target_act(self, states):
        """get target network actions from all the agents """
        target_actions = []
        for agent,state in zip(self.maddpg_agents,states):
            target_actions.append(agent.target_act(state))
        return target_actions


    def _update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        states, actions, rewards, next_states, dones = samples

        if self.config.log:
            print('\n')
            print('states',len(states),states[0].shape)
            print('actions',len(actions),actions[0].shape)
            print('rewards',len(rewards),rewards[0].shape)
            print('next states', len(next_states),next_states[0].shape)
            print('dones',len(dones),dones[0].shape)
            print('\n')

        current_agent = self.maddpg_agents[agent_number]

        current_agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.agents_target_act(next_states)

        if self.config.log:
            print('\ntarget_actions',len(target_actions),target_actions[0].shape)
            print('nextstates',len(next_states),next_states[0].shape)
            print('target_actions',len(target_actions),target_actions[0].shape)

        target_critic_input = torch.cat((*next_states, *target_actions), dim=1)

        if self.config.log:
            print('\ntarget_actions cat', target_critic_input.shape)

        with torch.no_grad():
            q_next = current_agent.target_critic(target_critic_input)

        target_q = rewards[agent_number].view(-1, 1) + self.config.maddpa_gamma * q_next * (1 - dones[agent_number].view(-1, 1))

        critic_input = torch.cat((*states, *actions), dim=1)

        estimated_q = current_agent.critic(critic_input)

        #huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = MSELoss()(estimated_q, target_q.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.critic.parameters(), self.config.grad_normalization_critic)
        current_agent.critic_optimizer.step()

        # update actor network using policy gradient
        current_agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative

        all_actor_actions = []
        for i,state in zip(range(self.num_agents),states):
            if i == agent_number:
                all_actor_actions.append(current_agent.actor(state))
            else:
                other_agent = self.maddpg_agents[i]
                all_actor_actions.append(other_agent.actor(state))
        if self.config.log:
            print(10*'#')
            print(all_actor_actions[0].shape)
            print(10 * '#')

        critic_input = torch.cat((*states, *all_actor_actions), dim=1)
        # get the policy gradient
        actor_loss = -current_agent.critic(critic_input).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        current_agent.actor_optimizer.step()

    def reset_noise(self):
        for agent in self.maddpg_agents:
            agent.reset_noise()

    def step(self, states):
        return [a.step(state) for a, state in zip(self.maddpg_agents, states)]

    def learn(self):
        if len(self.buffer) < self.config.maddpa_batch_size:
            return
        for _ in range(0,self.config.maddpa_n_learn_updates):
            for i_agent in range(len(self.maddpg_agents)):
                sample = self.buffer.sample()
                self._update(sample, i_agent)


        self.update_targets()

    def remember(self, states, actions, rewards, next_states, dones):

        self.buffer.append(states, actions, rewards, next_states, dones)



    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.config.maddpa_tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.config.maddpa_tau)







