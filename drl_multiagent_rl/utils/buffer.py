from collections import namedtuple, deque

import numpy as np
import random
import torch
from utils.agent_utilities import to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# few modifications from from https://github.com/RitwikSaikia/drlnd_p3_colab_compete/blob/master/replay_buffer.py

class ReplayBuffer(object):

    def __init__(self,buffer_size, batch_size, num_agents,seed):
        self.max_size = buffer_size
        self.num_agents = num_agents
        self.storage = [deque(maxlen=buffer_size) for _ in range(num_agents)]
        self.batch_size=batch_size
        self.seed = random.seed(seed)

        self.length = 0

    def __len__(self):
        return self.length

    def append(self, states, actions, rewards, next_states, dones):
        for i_agent in range(self.num_agents):
            e = Experience(np.float64(states[i_agent]),
                           np.float64(actions[i_agent]),
                           np.float64(rewards[i_agent]),
                           np.float64(next_states[i_agent]),
                           np.float64(dones[i_agent]))
            self.storage[i_agent].append(e)
            self.length = len(self.storage[i_agent])

    def sample(self):
        idxs = np.random.choice(np.arange(self.length), size=self.batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        eps = np.finfo(float).eps

        for i_agent in range(self.num_agents):
            all_experiences = self.storage[i_agent]
            batch_experiences = [all_experiences[i] for i in idxs]
            states.append(to_tensor([e.state for e in batch_experiences]).float().to(device))
            actions.append(to_tensor([e.action for e in batch_experiences]).float().to(device))
            next_states.append(to_tensor([e.next_state for e in batch_experiences]).float().to(device))
            dones.append(to_tensor([e.done for e in batch_experiences]).float().to(device))

            batch_rewards = np.float64([e.reward for e in batch_experiences])
            all_rewards = np.float64([e.reward for e in all_experiences])

            # Normalize for better performance
            batch_rewards = (batch_rewards - all_rewards.mean()) / (all_rewards.std() + eps)

            rewards.append(to_tensor(batch_rewards).float().to(device))

        return states, actions, rewards, next_states, dones


Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))
