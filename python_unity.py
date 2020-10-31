'''
Autonomous Driving Car simulation with Unity Enviroment.

Author : Shinhyeok Hwang, Yoojin Cho
Course : CoE202
Algorithm : DDPG(Deep Deterministic Policy Gradient)
https://arxiv.org/pdf/1509.02971.pdf
'''

import math
import random
import numpy as np
from copy import copy, deepcopy

import torch

from ddpg_agent import Agent
from mlagents_envs.environment import UnityEnvironment


#Hyperparameters
MAX_EPISODES=1000   #number of episodes of the training
MAX_STEPS=200   #max steps to finish an episode. An episode breaks early if some break conditions are met (like too much
                #amplitude of the joints angles or if a failure occurs). In the case of pendulum there is no break
                #condition, hence no environment reset,  so we just put 1 step per episode. 
epsilon = 1
epsilon_decay = 1./20000 #this is ok for a simple task like inverted pendulum, but maybe this would be set to zero for more
                #complex tasks like Hopper; epsilon is a decay for the exploration and noise applied to the action is 
                #weighted by this decay. In more complex tasks we need the exploration to not vanish so we set the decay
                #to zero.
PRINT_EVERY = 10    #Print info about average reward every PRINT_EVERY

#set GPU for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set unity environment
env = UnityEnvironment(file_name = 'Road1/Prototype 1')
env.reset()

behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

# x, y, z = cur_obs[0:3], x', y', z' = cur_obs[3:6]
# s1, s2, s3, s4, s5 = cur_obs[6:11]
state = cur_obs[6:11]
action_angle = 0
action_torque = [150, 150]

agents = Agent(state_dim=3, action_dim=1, device=device)

best_reward = -np.inf
saved_reward = -np.inf
saved_ep = 0
average_reward = 0
global_step = 0

for episode in range(MAX_EPISODES):

    agents.reset()

    ep_reward = 0

    #Receive Initial Observation state.
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    state = cur_obs[6:11]

    for timestep in range(MAX_STEPS):

        global_step += 1
        #epsilon -= epsilon_decay

        #Select action a_t = Policy + Noise
        action_angle = agents.get_action(state)

        #Execute action a_t & Move the simulation forward
        env.set_actions(behavior_name, np.array([[action_angle, 150, 150]]))
        env.step()

        #Observe Observe reward r_t and new state s_(t+1)
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        cur_obs = decision_steps.obs[0][0,:]
        next_state = cur_obs[6:11]
        reward = 1      #need to fix
        terminal = 0    #see if episode is finished

        #Store Transition in memory R
        agents.store_to_memory(state, action_angle, reward, terminal, next_state)
        
        #train models
        agents.train(device)

        ep_reward += reward

        if terminal:
            agents.reset()
            break
    
    if ep_reward > best_reward:
        #Save the actor model for future testing
        torch.save(agents.actor.state_dict(), 'best_model.pkl')
        best_reward = ep_reward
        saved_reward = ep_reward
        saved_ep = episode + 1

    #Print every print_every episodes
    if (episode % PRINT_EVERY) == (PRINT_EVERY-1):
        #subplot(plot_reward, plot_policy, plot_q, plot_steps)
        print('[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'.format(PRINT_EVERY) %
              (episode + 1, global_step, average_reward / PRINT_EVERY))
        print("Last model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
        average_reward = 0 #reset average reward

env.close()