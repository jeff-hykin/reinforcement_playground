from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch
from super_map import LazyDict
import math
from collections import defaultdict
import functools
from gym.wrappers import AtariPreprocessing

from prefabs.baselines_optimizer import RMSpropTFLike

import tools.stat_tools as stat_tools
from tools.basics import product, flatten
from tools.stat_tools import rolling_average
from tools.basics import product, flatten
from tools.debug import debug
from tools.pytorch_tools import Network, layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential

from agent_builders.my_small_a2c_vectorized.main import Agent

env = gym.make("CartPole-v1")
mr_bond = Agent(
    observation_space=env.observation_space,
    action_space=env.action_space
)
mr_bond.when_mission_starts()
for episode_index in range(500):
    mr_bond.episode_is_over = False
    mr_bond.observation = env.reset()
    mr_bond.when_episode_starts(episode_index)
    
    timestep_index = -1
    while not mr_bond.episode_is_over:
        timestep_index += 1
        
        mr_bond.when_timestep_starts(timestep_index)
        mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
        mr_bond.when_timestep_ends(timestep_index)
            
    mr_bond.when_episode_ends(episode_index)
mr_bond.when_mission_ends()
env.close()