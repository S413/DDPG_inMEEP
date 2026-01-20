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

# TODO: create here the variables to keep track of best designs, e.g. best total transmission, best balanced transmission, etc.
# TODO: create here reward functions as well; expect large implementation differences for objective split ratio

class RewardHelper:
    def __init__(self, template: str,
                 best_T: float = 0.6,
                 delta_T: float = 0.06, 
                 target_ratio: float = 3.0,
                 delta: float = 0.2):
        '''
        A helper to select the reward function to use, and what parameters to 
        keep track of.
        Could be extended to have more options as templates increase. Consolidate later.
        '''
        self.template = template
        
        if template == "2x2":
            self.best_T = best_T # since this is roughly transmission of opposite output arm
            self.delta_T = delta_T # roughly 10%
        else:
            self.target_ratio = target_ratio # target split ratio
            self.delta = delta      # acceptable deviation from target ratio
            self.total_trans = 0.0

    def compute_reward(self, Tt, Tb):
        if self.template == "2x2":
            return reward_balanced_transmission(Tt, Tb)
        else:
            return reward_mini(Tt, Tb, self.delta, self.target_ratio)
        
    def update_best_params(self, Tt, Tb):
        if self.template == "2x2":
            # Update best_T based on current performance
            current_total = torch.min(Tt) + torch.min(Tb)
            if current_total > self.best_T:
                self.best_T = current_total
            if abs(torch.min(Tt) - torch.min(Tb)) < self.delta_T:
                self.delta_T = abs(torch.min(Tt) - torch.min(Tb))

        else:
            # for now we just keep track of the total transmission since the
            # different splits make it a bit more code heavy to track ratios
            self.total_trans = torch.min(Tt) + torch.min(Tb)

    def checking_criteria(self, Tt, Tb):
        if self.template == "2x2":
            if (torch.abs(Tt - Tb) <= self.delta_T) and (Tt+Tb >= self.best_T):
                self.update_best_params(Tt, Tb)
                return self.compute_reward(Tt, Tb) + 10.0, True # bonus for meeting criteria, flag on
            else:
                return self.compute_reward(Tt, Tb), False # no bonus, flag off
        else:
            # for mini splitter, just positive reward is delta within target ratio
            reward = self.compute_reward(Tt, Tb)
            if reward > 0:
                return reward, True 
            else:
                return reward, False
            

def reward_balanced_transmission(Tt, Tb, best_T):
    reward = (
                -best_T + torch.min(Tt)+torch.min(Tb) + # if total transmission is high, difference will be 0 or +
                -2.0 * torch.abs(torch.min(Tt)-torch.min(Tb)) # large difference in transission per arm is bad
                )
    
    return reward

def reward_mini(Tt, Tb, delta, ratio):
    '''
    Reward function for the 1x2 splitter.
    Taken from the paper with minor modifications (penalization for termination,
    because they are not a possible action in our more constrained setup)
    '''
    if torch.max(torch.min(Tt), torch.min(Tb)) >= 0.5:
        if torch.abs(torch.mean(Tt / Tb) - ratio) < delta:
            reward = torch.mean(Tt / Tb)
        elif torch.abs(torch.mean(Tb / Tt) - ratio) < delta:
            reward = torch.mean(Tb / Tt)
        else:
            reward = torch.max(torch.min(Tb / Tt), torch.min(Tt / Tb))
    else:
        reward = -10
    
    return reward

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
            
    
    def load(self, path='./ReplayBuffer/buffer_v2.pickle'):
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
