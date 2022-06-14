import torch
import torch.nn as nn
import random
from tqdm import tqdm
import pickle 
import gym
import numpy as np
import collections 
import cv2
import time

from tools.agent_skeleton import Skeleton
from tools.file_system_tools import FileSystem

class Agent(Skeleton):
    def __init__(self, 
        observation_space,
        action_space,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.001,
    ):
        self.observation_space = observation_space
        self.action_space      = action_space
        self.learning_rate     = learning_rate  
        self.discount_factor   = discount_factor
        self.epsilon           = epsilon        # Amount of randomness in the action selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        pass
    
    def when_mission_starts(self, mission_index=0):
        self.qtable = np.zeros((self.observation_space.n, self.action_space.n))
        self.outcomes = []
        pass
        
    def when_episode_starts(self, episode_index):
        self.discounted_reward_sum = 0
        pass
        
    def when_timestep_starts(self, timestep_index):
        self.prev_observation = self.observation
        # if random number < epsilon, take a random action
        if np.random.random() < self.epsilon:
            self.action = self.action_space.sample()
        # else, take the action with the highest value in the current self.observation
        else:
            self.action = np.argmax(self.qtable[self.observation])
        
    def when_timestep_ends(self, timestep_index):
        old_q_value       = self.qtable[self.prev_observation, self.action]
        discounted_reward = self.reward + self.discount_factor * np.max(self.qtable[self.observation]) 
        self.discounted_reward_sum += discounted_reward
        
        # update q value
        self.qtable[self.prev_observation, self.action] = old_q_value + self.learning_rate * (discounted_reward - self.qtable[self.prev_observation, self.action])
        
    def when_episode_ends(self, episode_index):
        self.outcome.append(self.discounted_reward_sum)
        pass
        
    def when_mission_ends(self, mission_index=0):
        pass