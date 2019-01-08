
from framework.networks import Actor,Critic
from torch.optim import Adam
from utils.buffer import *
from utils.agent_utilities import *
import os
from utils.noise import OUNoise
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent():

    # variables shared across instance of this class
    local_critic = None
    target_critic = None
    critic_optimizer = None


    def __init__(self, state_size, action_size, config,id_agent):
        super(DDPGAgent, self).__init__()

        self.local_actor = Actor(state_size, action_size, config).to(device)
        self.target_actor = Actor(state_size, action_size, config).to(device)

        self.actor_optimizer = Adam(self.local_actor.parameters(), lr=config.actor_lr, weight_decay=config.actor_weight_decay)

        self.config = config

        self.noise = OUNoise(action_size,config.noise_seed, mu=self.config.noise_mu, theta=self.config.noise_theta,
                             sigma=self.config.noise_sigma)

        #id to identify the agent
        self.agent_id = id_agent

        # Initilise Class levell Critic Network
        if DDPGAgent.local_critic is None:
            DDPGAgent.local_critic = Critic(state_size, action_size, config).to(device)
        if DDPGAgent.target_critic is None:
            DDPGAgent.target_critic = Critic(state_size, action_size, config).to(device)
        if DDPGAgent.critic_optimizer is None:
            DDPGAgent.critic_optimizer = Adam(DDPGAgent.local_critic.parameters(), lr=config.critic_lr, weight_decay=config.critic_weight_decay)

        self.critic = DDPGAgent.local_critic
        self.target_critic = DDPGAgent.target_critic
        self.critic_optimizer =DDPGAgent.critic_optimizer




    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.local_critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), self.config.grad_normalization_actor)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.local_actor(states)
        actor_loss = -self.local_critic(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local_critic, self.target_critic, self.config.maddpa_tau)
        self.soft_update(self.local_actor, self.target_actor, self.config.maddpa_tau)

    def local_act(self, state,exploration=1,add_noise=True):
        self.local_actor.eval()
        with torch.no_grad():
            if add_noise:
                action = self.local_actor(state) + to_tensor(exploration * self.noise.sample())
            else:
                action = self.local_actor(state)
        self.local_actor.train()
        action = torch.clamp(action, -1, 1)
        return action

    def reset_noise(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def save(self, folder, prefix, version):
        save_dict = {
            'local_actor': self.local_actor.state_dict(),
            'local_critic': self.local_critic.state_dict(),
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
        self.local_actor.load_state_dict(save_dict['local_actor'])
        self.local_critic.load_state_dict(save_dict['local_critic'])
        self.target_actor.load_state_dict(save_dict['target_actor'])
        self.target_critic.load_state_dict(save_dict['target_critic'])
