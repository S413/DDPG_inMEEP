# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:27:41 2025

@author: sergi
"""

## some other functions for the RL setup

import random
import torch
import pickle
import numpy as np

def reward_balanced_transmission(Tt, Tb):
    imbalance = torch.abs(Tt - Tb) / (Tt + Tb + 1e-6)
    return 0.5 * (Tt + Tb) * (1 - imbalance)

def reward_min(Tt, Tb):
    '''
    Give reward based on the smallest arm transmission.
    Should suppress preferring keeping one transmission high. 
    '''
    return torch.min(Tt, Tb) # since these are inversely correlated

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    def save(self):
        '''
        Save the replay buffer. Might come in handy to have a set of transitions too.
        '''
        data_to_save = []
        for s, a, r, s_next, done in self.buffer:
            data_to_save.append({
                'state':s.cpu(),
                'action':a.cpu(),
                'reward':r,
                'next_state':s_next.cpu(),
                'done':done
                })
        try:
            with open("./ReplayBuffer/buffer.pickle", 'wb') as f:
                pickle.dump(data_to_save, f)
            print("Successfully saved replay buffer.")
        except:
            print("Failed to save the buffer. Probably something is not serializable.")
            
    
    def load(self, path='./ReplayBuffer/buffer.pickle'):
        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.buffer = []
        for item in loaded_data:
            self.buffer.append((
                item['state'],
                item['action'],
                item['reward'],
                item['next_state'],
                item['done']
                ))
        print("Loaded replay buffer.")

class PriorityReplayBuffer:
    def __init__(self, capacity, alpha=0.6, initial_mode='reward'):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = list()
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.mode = initial_mode
        self.min_reward = -0.5 # arbitrary for now, update at push
        
    def set_mode(self, mode):
        assert mode in ['reward', 'td_error']
        self.mode = mode
        
    def push(self, state, action, reward, next_state, done, priority=None):
        """
        Add a transition with optional custom priority.
        If priority is None, it will be set based on the current mode.
        """
        self.min_reward = min(self.min_reward, reward) # update min
        
        if priority is None:
            if self.mode == 'reward':
                priority =  self.min_reward - reward + 1e-5  # reward-based priority, epsilon to prevent 0 priority
            else:
                priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum() 
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, torch.tensor(weights, dtype=torch.float32) # truly need the rest?
    
    def update_priorities(self, indices, values, epsilon=1e-5):
        """
        Update priorities either based on td_error or reward.
        - if mode == 'td_error': values = td_errors
        - if mode == 'reward':   values = rewards
        """
        for idx, val in zip(indices, values):
            val = np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=0.0)
            if self.mode == 'td_error':
                self.priorities[idx] = abs(val) + epsilon
            else:
                self.min_reward = min(self.min_reward, val)
                self.priorities[idx] = self.min_reward - val + epsilon

            
    def __len__(self):
        return len(self.buffer)
    
    def save(self):
        data_to_save = []
        for s, a, r, s_next, done in self.buffer:
            data_to_save.append({
                'state':s.cpu(),
                'action':a.cpu(),
                'reward':r,
                'next_state':s_next.cpu(),
                'done':done
            })
        try:
            with open("./ReplayBuffer/buffer.pickle", 'wb') as f:
                pickle.dump(data_to_save, f)
            print("Successfully saved replay buffer.")
        except:
            print("Failed to save the buffer. Probably something is not serializable.")
    
# class GaussianNoise:
#     '''
#     This is for exploration under DDPG
#     '''
#     def __init__(self, mu=0.0, sigma=0.2):
#         self.mu, self.sigma = mu, sigma
#     def sample(self, shape):
        # return torch.normal(self.mu, self.sigma, size=shape)
        
## added decay for exploration
class GaussianNoise:
    def __init__(self, sigma_start=0.2, sigma_final=0.05, decay_steps=50_000):
        self.sigma_start = sigma_start
        self.sigma_final = sigma_final
        self.decay_rate  = (sigma_final / sigma_start) ** (1.0 / decay_steps)
        self.sigma = sigma_start

    def sample(self, shape, device=None):
        if device is None:
            device = 'cpu'
        n = torch.normal(0, self.sigma, size=shape, device=device)
        # exponential annealing
        self.sigma = max(self.sigma_final, self.sigma * self.decay_rate)
        return n

# OU-noise, used in Lunar Lander in several examples and class
class OUNoise:
    def __init__(self, shape, mu=0.0, theta=0.15, sigma=0.2, scale=1.0):
        self.shape = shape
        self.size = np.prod(shape)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.scale = scale
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.size) * self.mu 
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return torch.tensor(self.state * self.scale, dtype=torch.float32).reshape(self.shape)
    
# def noise_scale_scheduler(epochs, epoch, initial_scale):
#     '''
#     A function to help scale the noise level down (from exploration)
#     as the epochs go down
#     '''
#     return ((epochs - epoch) / epochs) * initial_scale

# more aggressive decay: using exponential decay. Might consider decaying sigma  and theta as well...
def noise_scale_scheduler(epoch, initial_scale=2.0, decay_rate=0.95, min_scale=0.05):
    """
    Exponentially decaying noise scale.
    decay_rate: e.g. 0.95 decays by 5% each epoch
    min_scale: minimum noise to allow continued mild exploration
    """
    return max(initial_scale * (decay_rate ** epoch), min_scale)
