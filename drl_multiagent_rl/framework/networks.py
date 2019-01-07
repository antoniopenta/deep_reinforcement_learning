import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, config):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            config (class): class with all the configuration

        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(config.actor_seed)
        self.fc1 = nn.Linear(state_size, config.actor_fc1_units)
        self.fc2 = nn.Linear(config.actor_fc1_units, config.actor_fc2_units)
        self.fc3 = nn.Linear(config.actor_fc2_units, action_size)
        self.reset_parameters()
        self.config = config

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.config.actor_non_linearity(self.fc1(state))
        x = self.config.actor_non_linearity(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size,config):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            config (class): class with all the configuration

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(config.critic_seed)
        self.fc1 = nn.Linear(state_size, config.critic_fc1_units)
        self.fc2 = nn.Linear(config.critic_fc1_units+action_size, config.critic_fc2_units)
        self.fc3 = nn.Linear(config.critic_fc2_units, 2)
        self.reset_parameters()
        self.config=config

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.config.critic_non_linearity(self.fc1(state))
        if self.config.log:
            print('state.shape,cricit',state.shape)
            print('action shape,cricit',action.shape)
        x = torch.cat((xs, action), dim=1)
        x = self.config.critic_non_linearity(self.fc2(x))
        return self.fc3(x)

