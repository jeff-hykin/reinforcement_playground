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
from super_map import LazyDict

from tools.agent_skeleton import Skeleton
from tools.file_system_tools import FileSystem

from tools.debug import debug
from tools.basics import sort_keys, randomly_pick_from

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier, all_argmax_coordinates
from trivial_torch_tools import Sequential, init, convert_each_arg, product
from trivial_torch_tools.generics import to_pure, flatten

torch.manual_seed(1)

class CriticNetwork(nn.Module):
    @init.to_device()
    def __init__(self, *, input_shape, output_shape, learning_rate=0.1):
        super(CriticNetwork, self).__init__()
        self.input_size = product(flatten(input_shape))
        
        self.layers = Sequential()
        self.layers.add_module('flatten', nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.layers.add_module('lstm', nn.LSTM(self.input_size, output_shape))
        self.layers.add_module('softmax', nn.Softmax(dim=0))
        
        self.loss_function = nn.MSELoss()
        self.optimizer = self.get_optimizer(learning_rate)
    
    def get_optimizer(self, learning_rate):
        return optim.SGD(self.parameters(), lr=learning_rate)
    
    def update_weights(self, input_value, ideal_output):
        self.zero_grad()
        input_value.requires_grad = True
        current_output = self.forward(input_value)
        loss = self.loss_function(current_output, ideal_output)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def predict(self, input_batch):
        with torch.no_grad():
            return self.forward(input_batch)
    
    def forward(self, input_batch):
        values = self.layers.flatten(input_batch)
        lstm_out, _ = self.layers.lstm(values)
        return self.layers.softmax(lstm_out) # alternative: F.log_softmax(lstm_out, dim=1)
        
    
class Agent(Skeleton):
    def __init__(self,
        observation_space,
        action_space,
        actions=None,
        training=True,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.0001,
        default_value_assumption=0,
        get_best_action=None,
    ):
        self.observation_space = observation_space
        self.action_space      = action_space
        self.learning_rate     = learning_rate  
        self.discount_factor   = discount_factor
        self.epsilon           = epsilon        # Amount of randomness in the action selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        self.actions           = OneHotifier(
            possible_values=(  actions or tuple(range(product(self.action_space.shape or (self.action_space.n,))))  ),
        )
        self.q_input_size      = product(self.observation_space.shape or (self.observation_space.n,))
        self.critic            = CriticNetwork(input_shape=self.q_input_size, output_shape=len(self.actions))
        # TODO: one-hot encode actions
        self._table                   = LazyDict()
        self.default_value_assumption = default_value_assumption
        self._get_best_action         = get_best_action
        self.training                 = training
        pass
    
    def observation_to_tensor(self, observation):
        return to_tensor(observation).flatten().to(self.critic.hardware)
    
    def create_q_input(self, observation):
        return to_tensor([self.observation_to_tensor(observation)])
    
    # TODO add python caching
    def value_of(self, observation, action):
        input_tensor = self.create_q_input(observation)
        action_onehot = self.critic.predict(input_tensor)[0] # first element because its a batch of size=1
        value_of_specific_action = action_onehot[self.actions.value_to_index(action)]
        result = to_pure(value_of_specific_action)
        position_coordinates = tuple(to_pure(
            all_argmax_coordinates(observation.position)[0]
        ))
        sort_keys(self._table)
        self._table[action, position_coordinates] = result
        return result
    
    def bellman_update(self, prev_observation, action, new_value):
        action_q_distribution = self.critic.predict(self.create_q_input(prev_observation))
        action_index = self.actions.value_to_index(action)
        action_q_distribution[0][action_index] = new_value
        return self.critic.update_weights(
            input_value=self.create_q_input(observation=prev_observation),
            ideal_output=action_q_distribution, # wrapped in list to create a batch of size 1
        )
    
    def get_best_action(self, observation):
        if isinstance(self.action_space, gym.spaces.Discrete):
            input_tensor = self.create_q_input(observation)
            action_q_distribution = self.critic.predict(input_tensor)[0] # first element because its a batch of size=1
            return self.actions.onehot_to_value(action_q_distribution)
        elif callable(self._get_best_action):
            return self._get_best_action(self)
        else:
            raise Exception(f'''\n\nThe agent {self.__class__.__name__} doesn't have a way to choose the best action, please pass the argument: \n    {self.__class__.__name__}(get_best_action=lambda self: do_something(self.observation))\n\n''')
        pass
    
    # 
    # mission hooks
    # 
    
    def when_mission_starts(self, mission_index=0):
        self.outcomes = []
        self.running_epsilon = self.epsilon if self.training else 0
        pass
        
    def when_episode_starts(self, episode_index):
        self.discounted_reward_sum = 0
        pass
        
    def when_timestep_starts(self, timestep_index):
        self.prev_observation = self.observation
        if random.random() < self.running_epsilon:
            self.action = randomly_pick_from(self.actions)
        # else, take the action with the highest value in the current self.observation
        else:
            self.action = self.get_best_action(observation=self.observation)
        pass
    
    def when_timestep_ends(self, timestep_index):
        old_q_value       = self.value_of(self.prev_observation, self.action)
        best_action       = self.get_best_action(self.observation)
        discounted_reward = self.reward + self.discount_factor * self.value_of(self.observation, best_action)
        self.discounted_reward_sum += discounted_reward
        
        if self.training:
            # update q value
            new_q_value = old_q_value + self.learning_rate * (discounted_reward - self.value_of(self.prev_observation, self.action))
            self.bellman_update(self.prev_observation, self.action, new_q_value)
        pass
        
    def when_episode_ends(self, episode_index):
        self.outcomes.append(self.discounted_reward_sum)
        self.running_epsilon *= (1-self.epsilon_decay)
        pass
        
    def when_mission_ends(self, mission_index=0):
        pass