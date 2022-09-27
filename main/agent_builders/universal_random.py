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

class Agent(Skeleton):
    @enhance_with(EpisodeEnhancement, LoggerEnhancement)
    def __init__(self,
        observation_space,
        reaction_space,
        random_seed=None,
    ):
        self.observation_space = observation_space
        self.observation_shape = self.observation_space.shape or (self.observation_space.n,)
        self.observation_size  = product(self.observation_shape)
        self.reaction_space    = reaction_space
        self.reaction_shape    = self.reaction_space.shape or (self.reaction_space.n,)
        self.reaction_size     = product(self.reaction_shape)
        self.random_seed       = random_seed or time.time()
        
    # 
    # mission hooks
    # 
    def when_mission_starts(self):
        pass
        
    def when_episode_starts(self):
        pass
        
    def when_timestep_starts(self):
        # 
        # decide on an action
        # 
        self.random_seed += 1
        _randomizer_state = random.getstate()
        random.seed(self.random_seed)
        try:
            self.timestep.reaction = self.reaction_space.sample()
        finally:
            # go back to whatever it used to be
            random.setstate(_randomizer_state)
        
        return self.timestep.reaction
    
    def when_timestep_ends(self):
        pass
        
    def when_episode_ends(self):
        pass
        
    def when_mission_ends(self):
        pass