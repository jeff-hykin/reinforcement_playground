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

from world_builders.frozen_lake.with_momentum import FrozenLakeEnv as Env
from agent_builders.dqn_primitive.main import Agent

from informative_iterator import ProgressBar

def run(number_of_episodes_for_training=1000, number_of_episodes_for_testing=100):
    env = Env()
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.001,
    )
    
    # 
    # training
    # 
    mr_bond.when_mission_starts()
    for progress, episode_index in ProgressBar(range(number_of_episodes_for_training)):
        
        mr_bond.observation     = env.reset()
        mr_bond.when_episode_starts(episode_index)
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            mr_bond.when_timestep_ends(timestep_index)
            
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    
    # 
    # testing
    # 
    mr_bond.when_mission_starts()
    for progress, episode_index in ProgressBar(range(number_of_episodes_for_testing)):
        
        mr_bond.observation     = env.reset()
        mr_bond.when_episode_starts(episode_index)
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            mr_bond.when_timestep_ends(timestep_index)
            
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()


run()