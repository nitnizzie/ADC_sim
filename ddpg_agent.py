'''
Autonomous Driving Car simulation with Unity Enviroment.

Author : Shinhyeok Hwang, Yoojin Cho
Course : CoE202
Algorithm : DDPG(Deep Deterministic Policy Gradient)
https://arxiv.org/pdf/1509.02971.pdf
'''

import random
import numpy as np
from copy import copy, deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Critic, Actor


#Hyperparameters for Learning
BUFFER_SIZE = 1000000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001 #Target Network HyperParameters Update rate
LR_ACTOR = 0.0001   #LEARNING RATE ACTOR
LR_CRITIC = 0.001   #LEARNING RATE CRITIC
H1 = 400    #neurons of 1st layers
H2 = 300    #neurons of 2nd layers

MAX_EPISODES=1000   #number of episodes of the training
MAX_STEPS=200   #max steps to finish an episode. An episode breaks early if some break conditions are met (like too much
                #amplitude of the joints angles or if a failure occurs). In the case of pendulum there is no break
                #condition, hence no environment reset,  so we just put 1 step per episode. 
buffer_start = 100  #initial warmup without training
epsilon = 1
epsilon_decay = 1./100000 #this is ok for a simple task like inverted pendulum, but maybe this would be set to zero for more
                    #complex tasks like Hopper; epsilon is a decay for the exploration and noise applied to the action is 
                    #weighted by this decay. In more complex tasks we need the exploration to not vanish so we set the decay
                    #to zero.
PRINT_EVERY = 10    #Print info about average reward every PRINT_EVERY

#set GPU for faster training
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class replayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp = 0
        self.buffer=deque()

    def add(self, s, a, r, t, s2):
        experience=(s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)

        s, a, r, t, s2 = map(np.stack, zip(*batch))

        return s, a, r, t, s2

    def clear(self):
        self.buffer = deque()
        self.num_exp=0


#Ornstein-UhlenBeck Noise for exploration
class OU_Noise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OU_NOISE(mu={}, sigma={})'.format(self.mu, self.sigma)

class Agent():
    def __init__(self, state_dim = 3, action_dim = 1):
        self.noise = OU_Noise(mu=np.zeros(action_dim))

        self.critic  = Critic(state_dim, action_dim, H1, H2).to(device)
        self.actor = Actor(state_dim, action_dim, H1, H2).to(device)

        self.target_critic  = Critic(state_dim, action_dim, H1, H2).to(device)
        self.target_actor = Actor(state_dim, action_dim, H1, H2).to(device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
    
q_optimizer  = optim.Adam(critic.parameters(),  lr=LR_CRITIC)#, weight_decay=0.01)
policy_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)

MSE = nn.MSELoss()

memory = replayBuffer(BUFFER_SIZE)
