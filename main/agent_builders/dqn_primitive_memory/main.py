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



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier
from trivial_torch_tools import Sequential, init, convert_each_arg, product, to_pure

torch.manual_seed(1)

class CriticNetwork(nn.Module):
    @init.hardware
    def __init__(self, *, input_shape, output_shape, learning_rate=0.1):
        super(CriticNetwork, self).__init__()
        self.layers = Sequential()
        self.layers.add_module('flatten', nn.Flatten(start_dim=input_shape, end_dim=-1)) # 1 => skip the first dimension because thats the batch dimension
        self.layers.add_module('lstm', nn.LSTM(product(input_shape), output_shape))
        self.layers.add_module('softmax', nn.Softmax(dim=0))
        
        self.loss_function = nn.NLLLoss()
        self.optimizer = self.get_optimizer(learning_rate)
    
    def get_optimizer(self, learning_rate):
        return optim.SGD(self.parameters(), lr=learning_rate)
    
    def update_weights(self, input_value, output_value):
        self.zero_grad()
        loss = self.loss_function(input_value, output_value)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def predict(self, input):
        with torch.no_grad():
            return self.forward(inputs)
        
    
class Agent(Skeleton):
    def __init__(self, 
        observation_space,
        action_space,
        actions=None,
        training=True,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.001,
        default_value_assumption=0,
        get_best_action=None,
    ):
        self.observation_space = observation_space
        self.action_space      = action_space
        self.learning_rate     = learning_rate  
        self.discount_factor   = discount_factor
        self.epsilon           = epsilon        # Amount of randomness in the action selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        self.actions           = actions or tuple(range(product(self.action_space.shape))))
        self.one_hotifier      = OneHotifier(possible_values=self.actions)
        self.q_input_shape     = len(actions) + product(self.observation_space.shape)
        self.critic            = CriticNetwork(input_shape=self.q_input_shape, output_shape=len(self.actions))
        # TODO: one-hot encode actions
        self._table                   = defaultdict(lambda: self.default_value_assumption)
        self.default_value_assumption = default_value_assumption
        self._get_best_action         = get_best_action
        self.training                 = training
        pass
    
    def action_to_tensor(self, action):
        return self.one_hotifier.to_one_hot(action).to(self.critic.hardware)
    
    def observation_to_tensor(self, observation):
        return to_tensor(observation).to(self.critic.hardware)
    
    def create_q_input(self, action, observation):
        return torch.cat((
            self.action_to_tensor(action),
            self.observation_to_tensor(observation),
        ))
    
    # TODO add python caching
    def value_of(self, observation, action):
        input_tensor = self.create_q_input(action, observation)
        action_values = self.critic.predict(input_tensor)
        action_index = self.actions.index(action)
        value_of_specific_action = action_values[action_index]
        return to_pure(value_of_specific_action)
    
    def bellman_update(self, prev_observation, action, new_value):
        return self.critic.update_weights(
            input_value=self.create_q_input(prev_observation),
            output_value=to_tensor(new_value).to(self.critic.hardware),
        )
    
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
        # if random number < epsilon, take a random action
        if random.random() < self.running_epsilon:
            self.action = self.action_space.sample()
        # else, take the action with the highest value in the current self.observation
        else:
            self.action = self.get_best_action(observation=self.observation)
        pass
    
    def when_timestep_ends(self, timestep_index):
        old_q_value       = self.value_of(self.prev_observation, self.action)
        discounted_reward = self.reward + self.discount_factor * self.get_best_action(self.observation)
        self.discounted_reward_sum += discounted_reward
        
        if self.training:
            # update q value
            new_q_value = old_q_value + self.learning_rate * (discounted_reward - self.value_of(self.prev_observation, self.action))
            self.bellman_update(self.prev_observation, self.action, new_q_value)
        pass
        
    def when_episode_ends(self, episode_index):
        self.outcomes.append(self.discounted_reward_sum)
        self.running_epsilon *= self.epsilon_decay
        pass
        
    def when_mission_ends(self, mission_index=0):
        pass