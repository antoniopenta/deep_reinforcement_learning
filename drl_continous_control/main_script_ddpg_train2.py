
from unityagents import UnityEnvironment
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt
from framework.ddpg_agent import Agent
import torch


def ddpg(n_episodes=300, max_t=1000):
    """ Deep Deterministic Policy Gradients
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores_window = deque(maxlen=100)
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
            # actions = [agents[i].act(states[i]) for i in range(num_agents)]
            actions = np.array([agents[i].act(states[i]) for i in range(num_agents)])
            #             if t == 0:
            #                 print("actions", actions)
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done

            for i in range(num_agents):
                agents[i].step(t, states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states
            scores += rewards
            if t % 20:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(t, np.mean(scores), np.min(scores), np.max(scores)), end="")
            if np.any(dones):
                break
        score = np.mean(scores)
        scores_window.append(score)  # save most recent score
        scores_episode.append(score)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)),
              end="\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(Agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(Agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores_episode


if __name__=='__main__':

    version = 2  # version (int): version number to save model check point and files

    env = UnityEnvironment(file_name=os.path.join('env', 'Reacher.app'))

    file_scores = os.path.join('data', 'scores_' + str(version) + '.txt')

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

    scores = ddpg()


    with open(file_scores, 'w') as fscores:
            fscores.write('\n'.join([str(item) for item in scores]))
