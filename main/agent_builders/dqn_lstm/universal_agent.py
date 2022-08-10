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
from copy import deepcopy

from blissful_basics import product, max_index, flatten
from super_hash import super_hash
from super_map import LazyDict

from tools.universe.agent import Skeleton, Enhancement, enhance_with
from tools.universe.timestep import Timestep
from tools.universe.enhancements.basic import EpisodeEnhancement, LoggerEnhancement
from tools.file_system_tools import FileSystem
from tools.stat_tools import normalize

from tools.debug import debug
from tools.basics import sort_keys, randomly_pick_from
from tools.object import Object, Options

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from prefabs.simple_lstm import SimpleLstm
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier, all_argmax_coordinates
from trivial_torch_tools import Sequential, init, convert_each_arg, product
from trivial_torch_tools.generics import to_pure, flatten

import warnings
warnings.filterwarnings('error')
torch.manual_seed(1)

class Decision:
    def __init__(self, timestep):
        self.position = tuple(to_pure(
            all_argmax_coordinates(timestep.observation.position)[0] # there can be many argmax's, but we know there will only ever be 1 for position
        ))
        self.reaction = timestep.reaction
    def __hash__(self):
        return hash(tuple((self.reaction, self.position)))
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"{self.reaction}".rjust(7)+f"{self.position}"
    def __lt__(self, other):
        return tuple((self.position, self.reaction)) < tuple((other.position, other.reaction))
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
        self.decision_count = LazyDict().setdefault(lambda *args: 0)
        self.q_value_per_decision   = LazyDict().setdefault(lambda *args: 0)
        self.decision_table         = LazyDict()
        self.reward_table           = LazyDict().setdefault(lambda *args: 0)
        original()
        
    def when_timestep_starts(self, original):
        sanity.when_start.observation = deepcopy(self.episode.timestep.observation.position.clone().detach())
        self.position = self.timestep.position = tuple(to_pure(
            all_argmax_coordinates(self.episode.timestep.observation.position)[0]
        ))
        assert self.position == self.get_position(self.episode.timestep)
        original()
        
    def when_timestep_ends(self, original):
        self.decision = Decision(self.episode.timestep)
        if self.following_policy:
            self.reward_table[self.decision] += self.episode.timestep.reward
            self.decision_count[self.decision] += 1
            sort_keys(self.reward_table)
            sort_keys(self.decision_count)
            # noramlized_values = [ round(each * 1000)/1000 for each in normalize(tuple(self.decision_count.values())) ]
            # self.decision_table = LazyDict({  each_key: each_value for each_key, each_value in zip(self.decision_count.keys(), noramlized_values)  })
            self.decision_table = LazyDict({
                each_decision : f"{decision_count:b}".rjust(18)+", # immediate reward per action: "+ f"{reward_total/decision_count:.5f}".rjust(9)
                    for each_decision, decision_count, reward_total in zip(self.decision_count.keys(), self.decision_count.values(), self.reward_table.values())
            })
        original()
    
class ValueCriticEnhancement(Enhancement):
    """
        requires:
            self.episode.timestep.index
            self.reactions # finite list of each possible reaction
        adds:
            self.value_of(observation, reaction, episode_timestep_index)
                returns a scalar
            self.bellman_update((observation, reaction, episode_timestep_index), new_value)
    """
    
    def when_mission_starts(self, original, ):
        observation_shape = self.observation_space.shape or (self.observation_space.n,)
        
        self._critic_cache = {}
        self._critic_pipeline = None
        self._critic = CriticNetwork(
            input_size=product(observation_shape),
            output_size=len(self.reactions),
            number_of_layers=2,
        )
        self._sum_table     = Object(Options(default=lambda each, self: 0))
        self._count_table   = Object(Options(default=lambda each, self: 0))
        self._ideal_table   = Object(Options(default=lambda each, self: 0))
        self._critic_table  = Object(Options(default=lambda each, self: 0))
        
        def value_of(timestep):
            return self._critic_cache[timestep.index][timestep.reaction]
        
        def _critic_update_pipeline(timestep):
            observation = to_tensor(timestep.observation).float()
            # save a copy of the hidden inputs on the timestep
            timestep.critic = Object(
                hidden_inputs= self._critic_pipeline.previous_hidden_values and (
                    self._critic_pipeline.previous_hidden_values[0].clone().detach(),
                    self._critic_pipeline.previous_hidden_values[1].clone().detach(),
                ),
                output=None,
            )
            timestep.critic.output = self._critic_pipeline(observation)
            self._critic_cache[timestep.index] = {
                each_reaction : each_value
                    for each_reaction, each_value in zip(self.reactions, timestep.critic.output)
            }
        
        # 
        # attch methods
        # 
        self._critic_update_pipeline = _critic_update_pipeline
        self.value_of                = value_of
        original()
        
    def when_episode_starts(self, original, ):
        self._critic_pipeline = self._critic.pipeline()
        original()
        # cache the values in a table to allow self.value_of() to be called multiple times
        # (because the normal way, e.g. on-demand, would screw up the pipeline order/sequence)
        self._critic_cache = {}
        # preemtively get the first one (must be done for eveything else to work)
        self._critic_update_pipeline(self.episode.next_timestep)
    
    def when_timestep_ends(self, original):
        # the next observation is available instantly, so also immediately cache it to make the result available
        self._critic_update_pipeline(self.episode.next_timestep)
        original()
        
        # 
        # update weights
        # 
        if True:
            timestep        = self.timestep
            updated_q_value = self.timestep.updated_q_value
            # logging
            self.q_value_per_decision[self.decision] = updated_q_value; sort_keys(self.q_value_per_decision)
            
            
            # 
            # get tensor form of choice (for the loss function)
            # 
            reaction_index = self.reactions.value_to_index(timestep.reaction)     # reaction_index is the action we want the loss to affect (so we only change that part of the tensor)
            ideal_output = list(to_pure(each) for each in timestep.critic.output) # get whatever the weights wouldve been
            ideal_output[reaction_index] = updated_q_value                        # replace this one weight in the copy
            ideal_output = to_tensor(ideal_output)                                 
            
            
            # 
            # replay the older timestep
            # 
            # (the critic update above has caused the pipeline to be on t+1, and we need to update the weights for t+0)
            self._critic.optimizer.zero_grad()
            self._critic_pipeline.previous_hidden_values = timestep.critic.hidden_inputs # t+0
            timestep.critic.output = self._critic_pipeline( # replay the observation and hidden inputs to get the normal t+0 output with gradient tracking
                to_tensor(timestep.observation).float().requires_grad_(True)
            )
            
            # 
            # calculate average ideal update values for debugging
            # 
            self._sum_table[  self.position] += ideal_output
            self._count_table[self.position] += 1
            average_ideal = self._sum_table[self.position]/self._count_table[self.position] # NOTE: debugging only, averaging ideal defeats the ability of memory
            
            # 
            # actually update the weights
            # 
            loss = self._critic.loss_function(timestep.critic.output, average_ideal)
            loss.backward()
            self._critic.optimizer.step()
            
            # logging
            self._critic_table[self.position] = [ f"{each:.3f}".rjust(7) for each in to_pure(timestep.critic.output.clone().detach())] ; sort_keys(self._critic_table)
            self._ideal_table[ self.position] = [ f"{each:.3f}".rjust(7) for each in to_pure(average_ideal.clone().detach())]          ; sort_keys(self._ideal_table)
            
            # go back to the t+1 state
            assert self.next_timestep.index == timestep.index + 1
            self._critic_update_pipeline(self.next_timestep)
    
sanity = LazyDict(
    when_start=LazyDict(
        picked_left=False,
    ),
)
class QTableEnhancement(Enhancement):
    """
        requires:
            self.episode.timestep.index
            self.reactions # finite list of each possible reaction
        adds:
            self.value_of(observation, reaction, episode_timestep_index)
                returns a scalar
            self.bellman_update((observation, reaction, episode_timestep_index), new_value)
    """
    
    def when_mission_starts(self, original, ):
        self.q_table = LazyDict()
        
        def value_of(timestep):
            position_key = self.get_position(timestep)
            # position_key = hash(tuple(flatten(to_pure(timestep.observation))))
            if position_key not in self.q_table:
                self.q_table[position_key] = LazyDict()
            if timestep.reaction not in self.q_table[position_key]:
                return 0
            else:
                return self.q_table[position_key][timestep.reaction]
        
        # 
        # attch methods
        # 
        self.value_of       = value_of
        original()
    
    def when_timestep_ends(self, original):
        original()
        timestep = self.episode.timestep
        
        assert torch.all(sanity.observation == deepcopy(timestep.observation.clone().detach()))
        assert sanity.reaction == deepcopy(timestep.reaction)
        assert self.position == self.get_position(timestep)
        
        position_key = self.get_position(timestep)
        if position_key not in self.q_table:
            self.q_table[position_key] = {}
        
        self.q_table[position_key][timestep.reaction] = timestep.updated_q_value
        debug.sanity_q_table = f"position_key={position_key}, reaction={timestep.reaction}, value={timestep.updated_q_value}"
        compare_string       = f"position_key={position_key}, reaction={timestep.reaction}"
        self.debug.q_table = self.q_table
        
class Agent(Skeleton):
    @enhance_with(EpisodeEnhancement, LoggerEnhancement, FightFireEnhancement, QTableEnhancement)
    def __init__(self,
        observation_space,
        reaction_space,
        reactions=None,
        training=True,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.0001,
        default_value_assumption=0,
        get_greedy_reaction=None,
        random_seed=None,
    ):
        self.observation_space = observation_space
        self.observation_shape = self.observation_space.shape or (self.observation_space.n,)
        self.observation_size  = product(self.observation_shape)
        self.reaction_space    = reaction_space
        self.reaction_shape    = self.reaction_space.shape or (self.reaction_space.n,)
        self.reaction_size     = product(self.reaction_shape)
        self.learning_rate     = learning_rate
        self.discount_factor   = discount_factor
        self.reactions         = OneHotifier(
            possible_values=(  reactions or tuple(range(self.reaction_size))  ),
        )
        self.random_seed       = random_seed or time.time()
        self.training          = training
        self.epsilon           = epsilon        # Amount of randomness in the reaction selection
        self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
        self.debug             = LazyDict()
        
        self.default_value_assumption = default_value_assumption
        self._get_greedy_reaction       = get_greedy_reaction
    
    def get_position(self, timestep):
        for row_index, each_row in enumerate(timestep.observation.position):
            for column_index, each_cell in enumerate(each_row):
                if each_cell:
                    return row_index, column_index
    
    def get_greedy_reaction(self, timestep):
        import math
        observation       = timestep.observation
        reaction_values   = []
        
        max_value         = -math.inf
        greedy_reaction = None
        for each_reaction in self.reactions:
            value = self.value_of(
                Timestep(timestep, reaction=each_reaction)
            )
            self.debug[each_reaction] = value
            if value > max_value:
                max_value       = value
                greedy_reaction = each_reaction
        
        self.debug.best_action = greedy_reaction
        return greedy_reaction
    
    # 
    # mission hooks
    # 
    
    def when_mission_starts(self):
        self.discounted_rewards = []
        self.running_epsilon = self.epsilon if self.training else 0
        self.following_policy = None
        
    def when_episode_starts(self):
        self.discounted_reward_sum = 0
        
    def when_timestep_starts(self):
        # 
        # decide on an action
        # 
        self.random_seed += 1
        random.seed(self.random_seed)
        self.following_policy = random.random() > self.running_epsilon
        self.timestep.reaction = randomly_pick_from(self.reactions)
        random.seed(time.time()) # go back to actual random for other things
        
        # if not self.following_policy:
        #     self.timestep.reaction = randomly_pick_from(self.reactions)
        # # else, take the reaction with the highest value in the current self.observation
        # elif not self.timestep.reaction: # self.next_timestep.reaction may have already been calculated, 
        #     self.timestep.reaction = self.get_greedy_reaction(self.episode.timestep)
    
    def when_timestep_ends(self):
        assert torch.all(sanity.when_start.observation == self.timestep.observation.position)
        assert self.position == self.get_position(self.timestep)
        assert self.position == self.get_position(self.timestep)
        
        sanity.observation = deepcopy(self.timestep.observation.clone().detach())
        sanity.reaction = deepcopy(self.timestep.reaction)
        self.debug.sanity = f'''self.position = {self.position}, action = {self.timestep.reaction}, reward = {self.timestep.reward}'''
        
        self.next_timestep.reaction = self.get_greedy_reaction(self.episode.next_timestep)
        timestep      = self.episode.timestep
        next_timestep = self.episode.next_timestep
        q_value_current = to_pure(self.value_of(timestep))            # q_t0 = Q(s0, a0)
        q_value_next    = to_pure(self.value_of(next_timestep))       # q_t1 = Q(s1, a1)
        delta           = (self.discount_factor * q_value_next) - q_value_current  # delta = (gamma * q_t1) - q_t0
        
        # TODO: record discounted reward here
        
        timestep.updated_q_value = q_value_current + self.learning_rate * (timestep.reward + delta)
        
    def when_episode_ends(self):
        self.discounted_rewards.append(self.discounted_reward_sum)
        self.running_epsilon *= (1-self.epsilon_decay)
        pass
        
    def when_mission_ends(self):
        pass