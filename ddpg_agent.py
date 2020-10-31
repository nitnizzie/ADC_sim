'''
Autonomous Driving Car simulation with Unity Enviroment.

Author : Shinhyeok Hwang, Yoojin Cho
Course : CoE202
Algorithm : DDPG(Deep Deterministic Policy Gradient)
https://arxiv.org/pdf/1509.02971.pdf
'''

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

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

buffer_start = 100  #initial warmup without training


class replayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp = 0
        self.buffer=deque()

    def add(self, state, action, reward, terminal, next_state):
        experience=(state, action, reward, terminal, next_state)
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

        state, action, reward, terminal, next_state = map(np.stack, zip(*batch))

        return state, action, reward, terminal, next_state

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
    def __init__(self, state_dim = 5, action_dim = 1, device = 'cpu'):

        #Randomly Initialize Critic & Actor Network.
        self.critic  = Critic(state_dim, action_dim, H1, H2).to(device)
        self.actor = Actor(state_dim, action_dim, H1, H2).to(device)

        self.target_critic  = Critic(state_dim, action_dim, H1, H2).to(device)
        self.target_actor = Actor(state_dim, action_dim, H1, H2).to(device)

        #Initialize Target Neworks with weights.
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.q_optimizer  = optim.Adam(self.critic.parameters(),  lr=LR_CRITIC)#, weight_decay=0.01)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.loss = nn.MSELoss()

        #Initialize ReplayBuffer R
        self.memory = replayBuffer(BUFFER_SIZE)

        #Initialize random process N.
        self.noise = OU_Noise(mu=np.zeros(action_dim))

    def reset(self):
        self.noise.reset()

    def get_action(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        action = self.actor.get_action(state)
        if add_noise:
            action += self.noise()
        return np.clip(action, -1., 1.)
    
    def store_to_memory(self, state, action, reward, terminal, next_state):
        self.memory.add(state, action, reward, terminal, next_state)

    def train(self, device):
        #keep adding experiences to the memory until there are at least minibatch size samples
        if self.memory.count() <= buffer_start:
            return
    
        #Sample a Random-minibatch of N transitions from R
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.memory.sample(BATCH_SIZE)

        s_batch = torch.FloatTensor(s_batch).to(device)
        a_batch = torch.FloatTensor(a_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
        t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
        s2_batch = torch.FloatTensor(s2_batch).to(device)

        #compute loss for critic
        a2_batch = self.target_actor(s2_batch)
        target_q = self.target_critic(s2_batch, a2_batch) #detach to avoid updating target
        y = r_batch + (1.0 - t_batch) * GAMMA * target_q.detach()
        q = self.critic(s_batch, a_batch)

        #update critic
        self.q_optimizer.zero_grad()
        q_loss = self.loss(q, y) #detach to avoid updating target
        q_loss.backward()
        self.q_optimizer.step()

        #compute loss for actor & update actor
        self.policy_optimizer.zero_grad()
        policy_loss = -self.critic(s_batch, self.actor(s_batch)).mean()
        policy_loss.backward()
        self.policy_optimizer.step()
    
        #update target networks
        self.soft_update()

    def soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - TAU) + param.data * TAU
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - TAU) + param.data * TAU
            )
