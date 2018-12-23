import os
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np


from framework.model import *
from framework.dqn_agent import *








if __name__=='__main__':

    random_action = True

    seed = 12
    version = 1
    model_agent =   os.path.join('model','checkpoint_'+str(version)+'.pth')


    env = UnityEnvironment(file_name=os.path.join('env','Banana.app'))
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    #reset the env to get the number of states
    env_info = env.reset(train_mode=True)[brain_name]
    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    #instanziate the agent
    drop_p = 0.5
    #the drop_out is disable in this case with eval()

    agent = Agent(state_size=state_size, action_size=action_size, drop_p=drop_p,seed=seed,model=model_agent)
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        # the agent is using the predicted (best) prediction
        action = agent.act(state, 0.5)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        state = next_state
        score += reward
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

    env.close()


