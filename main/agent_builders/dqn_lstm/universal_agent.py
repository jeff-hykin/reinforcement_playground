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

from tools.universe.agent import Skeleton, Enhancement, enhance_with
from tools.universe.enhancements.basic import EpisodeEnhancement, LoggerEnhancement
from tools.file_system_tools import FileSystem
from tools.stat_tools import normalize

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

class Decision:
    def __init__(self, response, position):
        self.position = tuple(position)
        self.response = f"{response}" 
    def __hash__(self):
        return hash(tuple((self.response, self.position)))
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"{self.response}".rjust(7)+f"{self.position}"
    def __lt__(self, other):
        return tuple((self.position, self.response)) < tuple((other.position, other.response))
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
        self.loss_function = lambda current_output, ideal_output: mse_loss(current_output, ideal_output)
        self.optimizer = self.get_optimizer(learning_rate)
    
    def get_optimizer(self, learning_rate):
        return optim.SGD(self.parameters(), lr=learning_rate)
    
    def predict(self, input_batch):
        with torch.no_grad():
            return self.forward(input_batch)
        
    def pipeline(self):
        return self.layers.lstm.pipeline()

class FightFireEnhancement(Enhancement):
    """
        adds:
            self.decision_table
            self.reward_table
    """
    
    def when_mission_starts(self, original, ):
        # self.correct_decision_proportions = {
        #    DOWN(0, 0): 0, 
        #    LEFT(0, 0): 0, 
        #   RIGHT(0, 0): 1.0, 
        #      UP(0, 0): 0, 
        #    DOWN(1, 0): 0, 
        #    LEFT(1, 0): 0.5, 
        #   RIGHT(1, 0): 0.5, 
        #      UP(1, 0): 0, 
        #    DOWN(2, 0): 0, 
        #    LEFT(2, 0): 1.0, 
        #   RIGHT(2, 0): 0, 
        #      UP(2, 0): 0, 
        # }
        self._decision_table = LazyDict().setdefault(lambda *args: 0)
        self.decision_table = LazyDict()
        self.reward_table = LazyDict().setdefault(lambda *args: 0)
        original()
        
    def when_timestep_starts(self, original):
        self.position = tuple(to_pure(
            all_argmax_coordinates(self.timestep.observation.position)[0]
        ))
        original()
        
    def when_timestep_ends(self, original):
        decision = Decision(self.timestep.response, self.position)
        self.reward_table[decision] += self.timestep.reward
        self._decision_table[decision] += 1
        self.reward_table    = sort_keys(self.reward_table)
        self._decision_table = sort_keys(self._decision_table)
        # noramlized_values = [ round(each * 1000)/1000 for each in normalize(tuple(self._decision_table.values())) ]
        # self.decision_table = LazyDict({  each_key: each_value for each_key, each_value in zip(self._decision_table.keys(), noramlized_values)  })
        self.decision_table = LazyDict({
            each_decision : f"{decision_count}".rjust(9)+", # immediate reward total: "+ f"{reward_total}".rjust(9)
                for each_decision, decision_count, reward_total in zip(self._decision_table.keys(), self._decision_table.values(), self.reward_table.values())
        })
        original()
    
class ValueCriticEnhancement(Enhancement):
    """
        requires:
            self.episode.timestep.index
            self.responses # finite list of each possible response
        adds:
            self.value_of(observation, response, episode_timestep_index)
                returns a scalar
            self.bellman_update((observation, response, episode_timestep_index), new_value)
    """
    
    def when_mission_starts(self, original, ):
        observation_shape = self.observation_space.shape or (self.observation_space.n,)
        
        self._critic_table = {}
        self._critic_pipeline = None
        self._critic = CriticNetwork(
            input_size=product(observation_shape),
            output_size=len(self.responses),
            number_of_layers=2,
        )
        self._critic.optimizer.zero_grad()
        
        def value_of(observation, response, episode_timestep_index):
            response_index = self.responses.index(response)
            return self._critic_table[episode_timestep_index][response_index]
        
        def bellman_update(inputs, new_value):
            observation, response, episode_timestep_index = inputs
            response_index = self.responses.index(response)
            
            # this action is the one we want the loss to affect
            ideal_output = list(to_pure(each) for each in self._critic_pipeline.previous_output)
            ideal_output[response_index] = new_value
            ideal_output = to_tensor(ideal_output)
            
            # update weights
            loss = self._critic.loss_function(self._critic_pipeline.previous_output, ideal_output)
            loss.backward()
            self._critic.optimizer.step()
            self._critic.optimizer.zero_grad()
        
        def _critic_update_pipeline(index, observation):
            observation = to_tensor(observation).float()
            self._critic_table[index] = self._critic_pipeline(observation)
        
        # 
        # attch methods
        # 
        self._critic_update_pipeline = _critic_update_pipeline
        self.value_of                = value_of
        self.bellman_update          = bellman_update
        original()
        
    def when_episode_starts(self, original, ):
        self._critic_pipeline = self._critic.pipeline()
        original()
        # cache the values in a table to allow self.value_of() to be called multiple times
        # (because the normal way, e.g. on-demand, would screw up the pipeline order/sequence)
        self._critic_table = {}
        # preemtively get the first one
        self._critic_update_pipeline(self.next_timestep.index, self.next_timestep.observation)
    
    def when_timestep_ends(self, original):
        # the next observation is available instantly, so also immediately cache it
        self._critic_update_pipeline(self.next_timestep.index, self.next_timestep.observation)
        original()
    

class Agent(Skeleton):
    @enhance_with(EpisodeEnhancement, LoggerEnhancement, ValueCriticEnhancement, FightFireEnhancement)
    def __init__(self,
        observation_space,
        response_space,
        responses=None,
        training=True,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.0001,
        default_value_assumption=0,
        get_greedy_response=None,
        random_seed=None,
    ):
        self.observation_space = observation_space
        self.observation_shape = self.observation_space.shape or (self.observation_space.n,)
        self.observation_size  = product(self.observation_shape)
        self.response_space    = response_space
        self.response_shape    = self.response_space.shape or (self.response_space.n,)
        self.response_size     = product(self.response_shape)
        self.learning_rate     = learning_rate
        self.discount_factor   = discount_factor
        self.responses         = OneHotifier(
            possible_values=(  responses or tuple(range(self.response_size))  ),
        )
        self.random_seed       = random_seed or time.time()
        self.training          = training
        self.epsilon           = epsilon        # Amount of randomness in the response selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        
        self.default_value_assumption = default_value_assumption
        self._get_greedy_response       = get_greedy_response
    
    def get_greedy_response(self, observation):
        response_values = []
        for each_response in self.responses:
            response_values.append(
                self.value_of(self.timestep.observation, each_response, self.episode.timestep.index)
            )
        best_response = self.responses.onehot_to_value(response_values)
        return best_response
    
    # 
    # mission hooks
    # 
    
    def when_mission_starts(self):
        self.discounted_rewards = []
        self.running_epsilon = self.epsilon if self.training else 0
        
    def when_episode_starts(self):
        self.discounted_reward_sum = 0
        
    def when_timestep_starts(self):
        # 
        # decide on an action
        # 
        self.random_seed += 1
        random.seed(self.random_seed)
        if random.random() < self.running_epsilon:
            self.timestep.response = randomly_pick_from(self.responses)
        # else, take the response with the highest value in the current self.observation
        elif not self.timestep.response: # self.next_timestep.response may have already been calculated
            self.timestep.response = self.get_greedy_response(self.timestep.observation)
    
    def when_timestep_ends(self):
        self.next_timestep.response = self.get_greedy_response(self.next_timestep.observation)
        
        q_value_current = to_pure(self.value_of(self.timestep.observation     , self.timestep.response     , self.episode.timestep.index))  # q_t0 = Q(s0, a0)
        q_value_next    = to_pure(self.value_of(self.next_timestep.observation, self.next_timestep.response, self.episode.timestep.index))  # q_t1 = Q(s1, a1)
        delta           = (self.discount_factor * q_value_next) - q_value_current                                                           # delta = (gamma * q_t1) - q_t0
        
        # TODO: record discounted reward here
        
        if self.training:
            # update q value
            more_accurate_prev_q_value = q_value_current + self.learning_rate * (self.timestep.reward + delta)                             # q
            self.bellman_update(
                (self.timestep.observation, self.timestep.response, self.timestep.index),
                more_accurate_prev_q_value,
            )
        
    def when_episode_ends(self):
        self.discounted_rewards.append(self.discounted_reward_sum)
        self.running_epsilon *= (1-self.epsilon_decay)
        pass
        
    def when_mission_ends(self):
        pass