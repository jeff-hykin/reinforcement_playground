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
from collections import defaultdict

from blissful_basics import product, max_index
from super_hash import super_hash

from tools.agent_skeleton import Skeleton
from tools.file_system_tools import FileSystem

class Agent(Skeleton):
    def __init__(self, 
        observation_space,
        action_space,
        actions=None,
        value_of=None,
        training=True,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.001,
        default_value_assumption=0,
        bellman_update=None,
        get_best_action=None,
    ):
        self.observation_space = observation_space
        self.action_space      = action_space
        self.learning_rate     = learning_rate  
        self.discount_factor   = discount_factor
        self.epsilon           = epsilon        # Amount of randomness in the action selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        self.actions           = actions or tuple(range(product(self.action_space.shape)))
        self._table            = defaultdict(lambda: self.default_value_assumption)
        self.value_of          = value_of or (lambda state, action: self._table[super_hash((state, action))])
        self.bellman_update    = bellman_update if callable(bellman_update) else (lambda state, action, value: self._table.update({ super_hash((state, action)): value }) )
        self.default_value_assumption = default_value_assumption
        self._get_best_action  = get_best_action
        self.training          = training
        pass
    
    def when_mission_starts(self, mission_index=0):
        self.outcomes = []
        self.running_epsilon = self.epsilon if not self.training else 0
        pass
        
    def when_episode_starts(self, episode_index):
        self.discounted_reward_sum = 0
        pass
        
    def when_timestep_starts(self, timestep_index):
        self.prev_observation = self.observation
        # if random number < epsilon, take a random action
        if random.random() < self.running_epsilon:
            self.action = self.action_space.sample()
        # else, take the action with the highest value in the current self.observation
        else:
            self.action = self.get_best_action(observation=self.observation)
        pass
    
    def get_best_action(self, observation):
        if isinstance(self.action_space, gym.spaces.Discrete):
            values = tuple((self.value_of(observation, each_action) for each_action in self.actions))
            best_action_key = max_index(values)
            return self.actions[best_action_key]
        elif callable(self._get_best_action):
            return self._get_best_action(self)
        else:
            raise Exception(f'''\n\nThe agent {self.__class__.__name__} doesn't have a way to choose the best action, please pass the argument: \n    {self.__class__.__name__}(get_best_action=lambda self: do_something(self.observation))\n\n''')
        pass
            
    def when_timestep_ends(self, timestep_index):
        old_q_value       = self.value_of(self.prev_observation, self.action)
        discounted_reward = self.reward + self.discount_factor * self.get_best_action(self.observation)
        self.discounted_reward_sum += discounted_reward
        
        # update q value
        new_value = old_q_value + self.learning_rate * (discounted_reward - self.value_of(self.prev_observation, self.action))
        self.bellman_update(self.prev_observation, self.action, new_value)
        pass
        
    def when_episode_ends(self, episode_index):
        self.outcomes.append(self.discounted_reward_sum)
        self.running_epsilon *= self.epsilon_decay
        pass
        
    def when_mission_ends(self, mission_index=0):
        pass