
from framework.networks import Actor,Critic
from torch.optim import Adam
import torch
import numpy as np

from utils.agent_utilities import *
import os

from utils.noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:


    critic_local = None
    critic_target = None
    critic_optimizer = None

    def __init__(self, state_size, action_size, config):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size, config).to(device)
        self.target_actor = Actor(state_size, action_size, config).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr, weight_decay=config.actor_weight_decay)

        self.config = config

        self.noise = OUNoise(2, mu=self.config.noise_mu, theta=self.config.noise_theta,
                             sigma=self.config.noise_sigma)


        # Initilise Class levell Critic Network
        if DDPGAgent.critic_local is None:
            DDPGAgent.critic_local = Critic(state_size, action_size,config).to(device)
        if DDPGAgent.critic_target is None:
            DDPGAgent.critic_target = Critic(state_size, action_size,config).to(device)
        if DDPGAgent.critic_optimizer is None:
            DDPGAgent.critic_optimizer = Adam(DDPGAgent.critic_local.parameters(), lr=config.critic_lr, weight_decay=config.critic_weight_decay)

        self.critic = DDPGAgent.critic_local
        self.target_critic = DDPGAgent.critic_target
        self.critic_optimizer =DDPGAgent.critic_optimizer

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)




    def step(self, obs,noise_scale):
        obs = obs.to(device)

        if np.random.random() > 0.5:
            delta = to_tensor(noise_scale * self.noise.sample())
        else:
            delta = -to_tensor(noise_scale * self.noise.sample())

        action = self.actor(obs) + delta

        action = torch.clamp(action, -1, 1)
        return action

    def target_act(self, obs):
        obs = obs.to(device)
        action = self.target_actor(obs)
        return action


    def reset_noise(self):
        self.noise.reset()

    def save(self, folder, prefix, version):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict()
        }
        filename = prefix+'_version_'+version+'.pth'
        filename = os.path.join(folder,filename)
        torch.save(save_dict, filename)

    def load(self, folder, prefix, version):
        filename = prefix + '_version_' + version + '.pth'
        filename = os.path.join(folder, filename)
        save_dict = torch.load(filename)
        self.actor.load_state_dict(save_dict['actor'])
        self.critic.load_state_dict(save_dict['critic'])
        self.target_actor.load_state_dict(save_dict['target_actor'])
        self.target_critic.load_state_dict(save_dict['target_critic'])

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()