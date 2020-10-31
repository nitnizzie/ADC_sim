'''
Autonomous Driving Car simulation with Unity Enviroment.

Author : Shinhyeok Hwang, Yoojin Cho
Course : CoE202
Algorithm : DDPG(Deep Deterministic Policy Gradient)
https://arxiv.org/pdf/1509.02971.pdf
'''

import random
import numpy as np
import torch

from ddpg_agent import Agent
from mlagents_envs.environment import UnityEnvironment


#set unity environment
env = UnityEnvironment(file_name = 'Road1/Prototype 1')
env.reset()

behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

# x, y, z = cur_obs[0:3], x', y', z' = cur_obs[3:6]
# s1, s2, s3, s4, s5 = cur_obs[6:11]
states = cur_obs[6:11]
action_angle = 0
action_torque = [150, 150]

reward = 0


for i in range(1000):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    states = cur_obs[6:11]

    # Set the actions
    env.set_actions(behavior_name, np.array([[0,150,150]]))
    # Move the simulation forward
    env.step()

env.close()