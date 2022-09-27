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

# a cheating agent thats really good at creating trajectories
class Agent(Skeleton):
    @enhance_with(EpisodeEnhancement, LoggerEnhancement,)
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
        self.has_water = None
        self.is_really_smart = randomly_pick_from([ True, False ])
        
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
        
        self.scenic_route_propensity = 0.4 # 0.4==40% of the time, if the first not-bad option is a scenic route, it'll take it
        
        self.default_value_assumption = default_value_assumption
        self._get_greedy_reaction       = get_greedy_reaction
    
    def get_position(self, timestep=None):
        timestep = timestep or self.timestep
        for row_index, each_row in enumerate(timestep.observation.position):
            for column_index, each_cell in enumerate(each_row):
                if each_cell:
                    return row_index, column_index
    
    def get_distance_between(self, new_position, ideal_positions=None):
        ideal_positions = self.ideal_positions if type(ideal_positions) == type(None)       else ideal_positions
            
        x, y = new_position
        minimum_distance = min(
            abs(x-good_x) + abs(y-good_y)
                for good_x, good_y in ideal_positions
        )
        return minimum_distance
    
    def predicted_position(self, action):
        x, y = self.get_position()
        if action == env.actions.LEFT:
            x -= 1
        elif action == env.actions.RIGHT:
            x += 1
        elif action == env.actions.DOWN:
            y -= 1
        elif action == env.actions.UP:
            y += 1
        else:
            raise Exception(f'''unknown action: {action}''')
        
        # stay in bounds
        if x > world.max_x_index: x = world.max_x_index
        if x < world.min_x_index: x = world.min_x_index
        if y > world.max_y_index: y = world.max_y_index
        if y < world.min_y_index: y = world.min_y_index
        
        return x, y
    # 
    # mission hooks
    # 
    
    def when_mission_starts(self):
        pass
        
    def when_episode_starts(self):
        self.has_water = False
        self.has_visited_fire = False
        self.has_almost_visted_water = False
        self.fire_coordinates = all_argmax_coordinates(world.state.grid.fire)
        self.water_coordinates = all_argmax_coordinates(world.state.grid.water)
        self.ideal_positions = self.water_coordinates
        
    def when_timestep_starts(self):
        current_position = self.get_position()
        # update flags
        if not self.has_water:
            self.has_water = 0 == self.get_distance_between(new_position=current_position, ideal_positions=self.water_coordinates)
        if not self.has_almost_visted_water:
            self.has_almost_visted_water = 1 == self.get_distance_between(new_position=current_position, ideal_positions=self.water_coordinates)
        if not self.has_visited_fire:
            self.has_visited_fire = 0 == self.get_distance_between(new_position=current_position, ideal_positions=self.fire_coordinates)
            
        # keep ideal coordiates up to date
        if self.is_really_smart:
            self.ideal_positions = self.fire_coordinates if self.has_water   else  self.water_coordinates
        else:
            if (self.has_almost_visted_water and not self.has_visited_fire) or self.has_water:
                self.ideal_positions = self.fire_coordinates
            else:
                self.ideal_positions = self.water_coordinates
        # 
        # pick action that doesn't hurt the distance to the nearest ideal position
        # 
        current_distance = self.get_distance_between(current_position)
        new_score = -math.inf
        possible_reactions = [ each for each in self.reactions ]
        random.shuffle(possible_reactions)
        fallback_option = possible_reactions[0]
        for each_reaction in possible_reactions:
            would_be_position = self.predicted_position(each_reaction)
            would_be_new_distance = self.get_distance_between(would_be_position)
            # skip all the definitely-bad options
            if would_be_new_distance > current_distance:
                continue
            # immediately take good options
            elif would_be_new_distance < current_distance:
                self.timestep.reaction = each_reaction
                return
            else:
                fallback_option = each_reaction
                # only occasionally confirm neutral options
                if random.random() < self.scenic_route_propensity:
                    self.timestep.reaction = each_reaction
                    return
        
        # if all of them were bad, just pick one
        self.timestep.reaction = fallback_option
        
    def when_timestep_ends(self):
        pass
        
    def when_episode_ends(self):
        pass
        
    def when_mission_ends(self):
        pass