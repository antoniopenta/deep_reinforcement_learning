import os
from collections import deque
import json
from unityagents import UnityEnvironment
import numpy as np
from utils.config import Config
from  framework.maddpg import MADDPGLearner
from framework.agent import DDPGAgent
from utils.agent_utilities import *



if __name__=='__main__':



    config = Config()

    Linux = False  #Linux (boolean): boolan value used to run on AWS (aws if Linux = True)

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



    all_scores = []

    scores = deque(maxlen=config.score_window_size)


    # each agent will take in  both the actions

    agents =  [DDPGAgent(state_size, action_size,config),
                             DDPGAgent(state_size, action_size,config)]

    maddpg = MADDPGLearner(agents, config)



    exploration_EPSILON_MAX = 1.0
    exploration_EPSILON_MIN = 0.05
    exploration_EPSILON_DECAY = 0.999
    exploration_eps= exploration_EPSILON_MAX

    for i_episode in range(0, config.num_episodes):

        env_info = env.reset(train_mode=True)[brain_name]

        states = env_info.vector_observations

        agent_scores_episode = np.zeros(maddpg.num_agents)

        maddpg.reset_noise()

        exploration_eps = max(exploration_EPSILON_MIN, exploration_eps * exploration_EPSILON_DECAY)

        score_episode = []

        for t_step in range(config.max_steps_4_episodes):

            torch_states = [to_tensor(states[i]) for i in range(maddpg.num_agents)]

            actions = maddpg.step(torch_states, noise_scale=exploration_eps)

            if config.log:
                print(100*'*')
                print('actions',actions)
                print('states',torch_states)
                print(100 * '*')

            actions = [action.data.numpy() for action in actions]

            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            #print('rewards',rewards,'t_stamp',t_step)
            dones = env_info.local_done
            if config.log:
                print('maddpg states',states.shape)
                print('maddpg next states',next_states.shape)
                print('maddpg rewards', len(rewards),rewards)

                print('maddpg dones', len(dones), dones)

            maddpg.remember(states, actions, rewards, next_states, dones)
            maddpg.learn()
            states = next_states
            agent_scores_episode += rewards

            score_episode.append(max(agent_scores_episode))



            if np.any(dones):
                #print(t_step)
                break
        value_score_episode = max(agent_scores_episode)
        all_scores.append(value_score_episode)
        scores.append(value_score_episode)
        avg_score = np.mean(scores)
        score_episode_mean=np.mean(score_episode)

        if i_episode % config.time_stamp_report:
            print(
                'Episode {}\t Average Score Last Espisode  : {:.4f}\t Average Score : {:.4f}\t Eps: {:.4} ,Length Episode {:4}\n'
                    .format(i_episode, score_episode_mean, avg_score,exploration_eps,t_step), end="")
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
            agent.save('model','checkpoint_'+str(i_episode - config.score_window_size), str(config.version))

    with open(file_scores, 'w') as fscores:
        fscores.write('\n'.join([str(item) for item in all_scores]))

    with open(file_version, 'w') as fversion:
        json.dump(config.getDict(), fversion)

    env.close()


