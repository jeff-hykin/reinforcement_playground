import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import collections 
import cv2
import time
from informative_iterator import ProgressBar
from statistics import mean as average

from world_builders.atari.custom_preprocessing import preprocess
from world_builders.atari.preprocessor_chao import TensorWrap
from agent_builders.auto_imitator.main import Agent

from tools.pytorch_tools import to_tensor


def default_mission(
        number_of_episodes=1000,
        env_name="BreakoutNoFrameskip-v4",
        frame_buffer_size=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    ):
    
    print('loading enviornment')
    env = preprocess(
        env=gym.make(env_name),
        frame_buffer_size=frame_buffer_size,
        frame_sample_rate=frame_sample_rate,
    )
    
    print('loading agent')
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_0.00021598702086765554.model",
    )
    
    print('starting mission')
    mr_bond.when_mission_starts()
    for progress, episode_index in ProgressBar(range(number_of_episodes)):
        print('episode_index = ', episode_index)
        if progress.updated and len(mr_bond.logging.episode_rewards) > 0:
            print("average reward: ", average(mr_bond.logging.episode_rewards))
        
        mr_bond.observation     = env.reset()
        mr_bond.reward          = 0
        mr_bond.episode_is_over = 0
        
        mr_bond.when_episode_starts(episode_index)
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            print('timestep_index = ', timestep_index, 'mr_bond.episode_is_over = ', mr_bond.episode_is_over, 'mr_bond.reward = ', mr_bond.reward)
            mr_bond.when_timestep_ends(timestep_index)
            
        print('calling when_episode_ends')
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    mr_bond.save()
    env.close()


default_mission(number_of_episodes=1000)