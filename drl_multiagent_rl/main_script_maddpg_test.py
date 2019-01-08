




import os
from collections import deque
import json
from unityagents import UnityEnvironment
from utils.config import Config
from framework.coordinator import *



if __name__=='__main__':

    config = Config()

    env = UnityEnvironment(file_name=os.path.join('env', 'Tennis.app'))

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

    # score buffers

    scores_window = deque(maxlen=config.score_window_size)
    scores = np.zeros(num_agents)
    scores_episode = []

    # agent

    agents = [DDPGAgent(state_size, action_size, config, 1),
              DDPGAgent(state_size, action_size, config, 2)]

    for agent in agents:
        agent.load('model','checkpoint_'+'agent_'+str(agent.agent_id)+'_226',str(config.version))

    scores = np.zeros(num_agents)
    try:
        while True:
            actions = []
            for agent, state in zip(agents, states):
                actions.append(agent.local_act(to_tensor(state),add_noise=False).data.numpy())

            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done

            states = next_states
            scores += rewards

            print('\rScores: {:.2f}\t{:.2f}'
                  .format(scores[0], scores[1]), end="")

            if np.any(dones):
                # reset the environment
                env_info = env.reset(train_mode=False)[brain_name]
                # examine the state space
                states = env_info.vector_observations
                pass
    except KeyboardInterrupt:
        print("\n Final Scores: {}".format(scores))

    env.close()

    print("\n Final Scores: {}".format(scores))