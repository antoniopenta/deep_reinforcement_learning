import torch.nn.functional as F

from utils.noise import *
import inspect
from types import FunctionType
import json

class Config():

    def __init__(self):

        self.version = 3  # version (int): version number to save model check point and files

        self.log = False # log used to print instructions

        self.report_value = 100
        self.time_stamp_report = 20  #

        self.num_episodes = 2000  # num_episodes (int): maximum number of training episodes
        self.max_score = 0.5  # max_score (float): that target score that we would like to reach, the benchmark is 15 in 1700 episode
        self.score_window_size = 100  # score_window_size(int): buffer size for the score
        self.max_steps_4_episodes = 1000  # max_t (int) max length of the time stemps


        self.actor_seed = 15
        self.critic_seed = 21

        self.actor_fc1_units = 256  # number neurons first layer actor
        self.actor_fc2_units = 128 # number neurons second layer actor
        self.actor_fc3_units = 64  # number neurons first layer actor

        self.critic_fc1_units = 256  # number neurons first layer critic
        self.critic_fc2_units = 128 # number neurons second layer critic
        self.critic_fc3_units= 64 # number neurons third layer critic

        self.critic_dropout_p = 0.5

        self.actor_non_linearity = F.relu
        self.actor_non_linearity.__name__='relu'


        self.critic_non_linearity = F.relu
        self.critic_non_linearity.__name__ = 'relu'



        self.actor_lr = 1e-4  # learning rate of the actor
        self.critic_lr = 1e-4  # learning rate of the critic

        self.actor_weight_decay = 0.0  # L2 weight decay
        self.critic_weight_decay = 0.0  # L2 weight decay


        self.noise_seed = 34
        self.noise_mu = 0.
        self.noise_theta = 0.15
        self.noise_sigma = 0.2

        self.agent_noise_multiplier = 1

        self.random_process_fn =  OUNoise(
            size=(2,), seed=self.noise_seed,mu=self.noise_mu,theta=self.noise_theta,sigma=self.noise_sigma)



        self.random_process_fn.__name__ = 'OUNoise'

        self.maddpa_buffer_size = int(1e6)  # replay buffer size
        self.maddpa_batch_size = 128  # minibatch size

        self.maddpa_n_learn_updates = 10  # number of learning updates
        self.maddpa_n_learn_steps = 20  # every n time step do update

        self.maddpa_gamma = 0.99  # discount factor
        self.maddpa_tau = 1e-3  # for soft update of target parameters

        self.buffer_seed = 28 # buffer seed for sampling


        self.grad_normalization_actor = 0.5

        self.grad_normalization_critic = 0.5


    def __str__(self):
        status = {}
        for x, y in self.__dict__.items():
            if type(y) == FunctionType:
                status[x] = getattr(self, x).__name__
            else:
                status[x] = str(getattr(self, x))
        return json.dumps(status)


    def getDict(self):
        status = {}
        for x, y in self.__dict__.items():
            if type(y) == FunctionType:
                status[x] = getattr(self, x).__name__
            else:
                status[x] = str(getattr(self, x))
        return status

if __name__=='__main__':

    config = Config()

    print(config)
    print(config.getDict())



