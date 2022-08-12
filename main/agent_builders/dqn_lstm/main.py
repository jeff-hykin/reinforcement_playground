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

from tools.agent_skeleton import Skeleton, enhance_with, AgentBasics, Enhancement
from tools.file_system_tools import FileSystem

from tools.debug import debug
from tools.basics import sort_keys, randomly_pick_from

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from prefabs.simple_lstm import SimpleLstm
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier, all_argmax_coordinates
from tools.timestep_tools import TimestepSeries, Timestep
from trivial_torch_tools import Sequential, init, convert_each_arg, product
from trivial_torch_tools.generics import to_pure, flatten

import warnings
warnings.filterwarnings('error')
torch.manual_seed(1)

# 
# Slider Enhancement
# 
class TimestepSeriesEnhancement(Enhancement):
    def when_episode_starts(self, original, *args):
        self.timestep_series = TimestepSeries()
        self.timestep_series[0].observation = self.observation
        original(*args)
    
    def when_timestep_starts(self, original, *args):
        self.timestep_series[self.episode.timestep.index].observation = self.observation
        original(*args)
        self.timestep_series[self.episode.timestep.index].action = self.action
    
    def when_timestep_ends(self, original, *args):
        self.timestep_series[self.episode.timestep.index-1].reward = self.reward
        original(*args)

class Decision:
    def __init__(self, action, observation):
        self.position = tuple(to_pure(
            all_argmax_coordinates(observation.position)[0]
        ))
        self.action = action
    def __hash__(self):
        return hash(tuple((self.action, self.position)))
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"{self.action}".rjust(7)+f"{self.position}"
    def __lt__(self, other):
        return tuple((self.position, self.action)) < tuple((other.position, other.action))
    def __eq__(self, other):
        return hash(self) == hash(other)

class CriticNetwork(nn.Module):
    @init.forward_sequential_method
    @init.to_device()
    def __init__(self, *, input_size, output_size, number_of_layers, learning_rate=0.1):
        super(CriticNetwork, self).__init__()
        
        self.layers = Sequential()
        self.layers.add_module('lstm', SimpleLstm(
            input_size=input_size,
            output_size=output_size,
            number_of_layers=number_of_layers,
        ))
        
        mse_loss = nn.MSELoss()
        self.loss_function = lambda current_output, ideal_output: (mse_loss(current_output, ideal_output)+10)**2
        self.optimizer = self.get_optimizer(learning_rate)
    
    def get_optimizer(self, learning_rate):
        return optim.SGD(self.parameters(), lr=learning_rate)
    
    def update_weights(self, input_value, ideal_output):
        self.optimizer.zero_grad()
        input_value.requires_grad = True
        current_output = self.forward(input_value)
        loss = self.loss_function(current_output, ideal_output)
        print(f'''loss = {loss}''')
        loss.backward()
        self.optimizer.step()
        return loss
    
    def predict(self, input_batch):
        with torch.no_grad():
            return self.forward(input_batch)

class ValueCriticEnhancement(Enhancement):
    """
        requires:
            self.episode.timestep.index
            self.actions # finite list of each possible action
        adds:
            self.value_of(observation, action, episode_timestep_index)
                returns a scalar
            self.bellman_update((observation, action, episode_timestep_index), new_value)
    """
    
    def when_mission_starts(self, original, *args):
        observation_shape = self.observation_space.shape or (self.observation_space.n,)
        
        self._critic_index = None
        self._critic_table = {}
        self._critic_pipeline = None
        self._critic = CriticNetwork(
            input_size=product(observation_shape),
            output_shape=len(self.actions),
        )
        
        def value_of(observation, action, episode_timestep_index):
            action_index = self.actions.index(action)
            return self._critic_table[episode_timestep_index][action_index]
        
        def bellman_update(inputs, new_value):
            observation, action, episode_timestep_index = inputs
            
            # update weights
            loss = self._critic.loss_function(self._critic_pipeline.previous_output, to_tensor(new_value))
            loss.backward()
            self._critic.optimizer.step()
            self._critic.optimizer.zero_grad()
        
        def _critic_update_pipeline(index, observation):
            observation = to_tensor(observation)
            observation.requires_grad = True
            self._critic_table[index] = self._critic_pipeline(observation)
        
        def update_weights(self, ideal_output):
            loss = self.loss_function(self._critic_pipeline.previous_output, ideal_output)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        
        self._critic_update_pipeline = _critic_update_pipeline
        self.value_of                = value_of
        self.bellman_update          = bellman_update
        original(*args)
        
    def when_episode_starts(self, original, *args):
        # create or reset the pipeline
        self._critic_pipeline = self._critic.pipeline()
        original(*args)
        # add the first observation
        # cache the values in a table to allow self.value_of() to be called multiple times
        # (the normal way of computing on-demand would screw up the pipeline order/sequence)
        self._critic_update_pipeline(self.episode.timestep.index, self.observation)
    
    def when_timestep_starts(self, original, *args):
        # record the time, whatever it may be
        self._critic_index = self.episode.timestep.index
    
    def when_timestep_ends(self, original, *args):
        # the next observation is available instantly, but because its the next observation we make sure the index is +1 of the previous index
        self._critic_update_pipeline(self._critic_index+1, self.observation)
        original(*args)
    

class Agent(Skeleton):
    @enhance_with(AgentBasics, TimestepSeriesEnhancement)
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
        get_greedy_action=None,
        random_seed=None,
    ):
        self.observation_space = observation_space
        self.action_space      = action_space
        self.learning_rate     = learning_rate  
        self.discount_factor   = discount_factor
        self.epsilon           = epsilon        # Amount of randomness in the action selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        self.sequence_size     = 5
        self.actions           = OneHotifier(
            possible_values=(  actions or tuple(range(product(self.action_space.shape or (self.action_space.n,))))  ),
        )
        self.q_input_size      = product(self.observation_space.shape or (self.observation_space.n,))
        
        # TODO: one-hot encode actions
        self._table                   = LazyDict()
        self.default_value_assumption = default_value_assumption
        self._get_greedy_action         = get_greedy_action
        self.training                 = training
        self.random_seed              = random_seed or time.time()
        if self.training:
            self.critic.train()
    
    @property
    def position(self):
        return tuple(to_pure(
            all_argmax_coordinates(self.observation.position)[0]
        ))
    
    def value_of(self, timestep_index, observation, action):
        
        # create critic input
        sequence = self.time_series[timestep_index-self.sequence_size:timestep_index]
        sequence[timestep_index].observation = observation # suppose what the current observation is
        sequence[timestep_index].action = action           # suppose what the current action is
        
        sequence_tensor = to_tensor([
            torch.cat(each_observation.flatten(), to_tensor(each_action))
                for each in sequence
        ])
        
        prediction = self.critic.predict(sequence_tensor)
        value_of_action = prediction[-1] # last layer is "current" prediction
        result = to_pure(value_of_action)
        self._table[Decision(action, observation)] = result
        sort_keys(self._table)
        return result
    
    def bellman_update(self, timestep_index, observation, action, new_value):
        # create critic input
        sequence = self.time_series[timestep_index-self.sequence_size:timestep_index]
        sequence[timestep_index].observation = observation # suppose what the current observation is
        sequence[timestep_index].action = action           # suppose what the current action is
        
        sequence_tensor = to_tensor([
            torch.cat(each_observation.flatten(), to_tensor(each_action))
                for each in sequence
        ])
        
        return self.critic.update_weights(
            input_value=sequence_tensor,
            ideal_output=new_value, # wrapped in list to create a batch of size 1
        )
    
    def get_greedy_action(self, observation):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_values = []
            for each_action in self.actions:
                action_values.append(
                    self.value_of(self.episode.timestep.index, self.observation, each_action)
                )
            best_action = self.actions.onehot_to_value(action_values)
            return best_action
        elif callable(self._get_greedy_action):
            return self._get_greedy_action(self)
        else:
            raise Exception(f'''\n\nThe agent {self.__class__.__name__} doesn't have a way to choose the best action, please pass the argument: \n    {self.__class__.__name__}(get_greedy_action=lambda self: do_something(self.observation))\n\n''')
        pass
    
    # 
    # mission hooks
    # 
    
    def when_mission_starts(self, mission_index=0):
        self.outcomes = []
        self.running_epsilon = self.epsilon if self.training else 0
        
    def when_episode_starts(self, episode_index):
        self.discounted_reward_sum = 0
        
    def when_timestep_starts(self, timestep_index):
        self.random_seed += 1
        random.seed(self.random_seed)
        if random.random() < self.running_epsilon:
            if self.position == (0,0):
                self.action = "RIGHT" # this most helpful action on a (3,1) field that I am currently debugging. Can safely be removed
            else:
                self.action = randomly_pick_from(self.actions)
        # else, take the action with the highest value in the current self.observation
        else:
            if self.action is None:
                self.action = self.get_greedy_action(observation=self.observation)
    
    def when_timestep_ends(self, timestep_index):
        self.action       = self.get_greedy_action(self.observation)
        q_value_previous  = self.value_of(self.episode.timestep.index-1, self.previous_observation, self.previous_observation_reaction)
        q_value_current   = self.value_of(self.episode.timestep.index  , self.observation         , self.action)
        delta             = (self.discount_factor * q_value_current) - q_value_previous
        
        discounted_reward = (self.reward + delta)
        self.discounted_reward_sum += discounted_reward # TODO: doesn't seem quite right to me
        
        if self.training:
            # update q value
            more_accurate_prev_q_value = q_value_previous + self.learning_rate * (self.reward + delta)
            self.bellman_update(self.episode.timestep.index-1, self.prev_observation, self.prev_observation_reaction, more_accurate_prev_q_value)
        
    def when_episode_ends(self, episode_index):
        self.outcomes.append(self.discounted_reward_sum)
        self.running_epsilon *= (1-self.epsilon_decay)
        pass
        
    def when_mission_ends(self, mission_index=0):
        pass