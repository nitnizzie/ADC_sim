'''
Autonomous Driving Car simulation with Unity Enviroment.

Author : Shinhyeok Hwang, Yoojin Cho
Course : CoE202
Algorithm : DDPG(Deep Deterministic Policy Gradient)
https://arxiv.org/pdf/1509.02971.pdf
'''

import numpy as np

import torch
import torch.nn as nn

#set GPU for faster training
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Network Architectures
def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, init_w=3e-3):
        super(Critic, self).__init__()
                
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        #self.bn1 = nn.BatchNorm1d(h1)
        
        self.linear2 = nn.Linear(h1+action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x,action],1))
        
        x = self.relu(x)
        x = self.linear3(x)
        
        return x
    

class Actor(nn.Module): 
    def __init__(self, state_dim, action_dim, h1=400, h2=300, init_w=0.003):
        super(Actor, self).__init__()
        
        #self.bn0 = nn.BatchNorm1d(state_dim)
        
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        #self.bn1 = nn.BatchNorm1d(h1)
        
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        #self.bn2 = nn.BatchNorm1d(h2)
        
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        #state = self.bn0(state)
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device) #(1, state_dim)
        self.eval()
        with torch.no_grad():
            action = self.forward(state)
        self.train()
        return action.detach().cpu().numpy()[0]