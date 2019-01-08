import os
from collections import deque
import json
from unityagents import UnityEnvironment
from utils.config import Config
from framework.coordinator import *


def exploration_strategy_1(eps, config):
    eps *= config.exploration_episilon_decay
    if eps < config.exploration_episilon_min:
        return config.exploration_episilon_min
    return eps


if __name__=='__main__':



    config = Config()

    Linux = True  #Linux (boolean): boolan value used to run on AWS (aws if Linux = True)

    file_scores = os.path.join('data','scores_'+str(config.version)+'.txt')

    file_version = os.path.join('data','version_'+str(config.version)+'.txt')

    if Linux:
        env = UnityEnvironment(file_name=os.path.join(os.getcwd(),'env', 'Tennis_Linux_NoVis','Tennis.x86_64'))
    else:
        env = UnityEnvironment(file_name=os.path.join('env','Tennis.app'))

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


    # score buffers

    scores_window = deque(maxlen=config.score_window_size)
    scores = np.zeros(num_agents)
    scores_episode = []

    # agent

    agents =  [DDPGAgent(state_size, action_size,config,1),
                             DDPGAgent(state_size, action_size,config,2)]

    # coordinator
    maddpg = MADDPGCoordinator(agents,action_size,config)

    exploration_eps = config.exploration_epsilon_max



    for i_episode in range(0, config.num_episodes):

        env_info = env.reset(train_mode=True)[brain_name]

        states = env_info.vector_observations

        maddpg.reset_noise()

        exploration_eps = config.exploration_epsilon_max

        scores = np.zeros(num_agents)

        for t_step in range(config.max_steps_4_episodes):

            # get the actions from the angent
            actions = maddpg.agents_act(states, exploration=exploration_eps)
            #run the actions
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            #get the result
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done
            #save the result in the buffer
            maddpg.remember(states, actions, rewards, next_states, dones)
            #make a step
            maddpg.step(t_step)
            # update sarsa
            states = next_states
            scores += rewards
            #reduce explortion
            exploration_eps = exploration_strategy_1(exploration_eps,config)

            if np.any(dones):
                #print(t_step)
                break

        score = np.max(scores)
        scores_window.append(score)  # save most recent score
        scores_episode.append(score)
        avg_score = np.mean(scores_window)

        if i_episode % config.time_stamp_report:
            print(
                'Episode {}\t Average Score : {:.4f}\t Eps: {:.4} ,Length Episode {:4}\n'
                    .format(i_episode, avg_score,exploration_eps,t_step), end="")
        if avg_score >= config.max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - config.score_window_size,
                avg_score))
            for index_agent,agent in enumerate(agents):
                agent.save('model','checkpoint_'+str(i_episode - config.score_window_size),config.version)



    if avg_score < config.max_score:
        print('\nEnvironment not solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
            i_episode - config.score_window_size,
            avg_score))
        for index_agent, agent in enumerate(agents):
            agent.save('model','checkpoint_'+'agent_'+str(index_agent)+'_'+str(i_episode - config.score_window_size), str(config.version))

    with open(file_scores, 'w') as fscores:
        fscores.write('\n'.join([str(item) for item in scores_episode]))

    with open(file_version, 'w') as fversion:
        json.dump(config.getDict(), fversion)

    env.close()


