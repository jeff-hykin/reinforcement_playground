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


from world_builders.atari.custom_preprocessing import preprocess, AutoLatentSpaceWrap, TensorWrap
from agent_builders.dqn_chao_latent.main import Agent

from informative_iterator import ProgressBar
from tools.pytorch_tools import to_tensor

# remaining: 12:32sec | [==>......................] 10.00% | 100/1000 | started: 07:07:24 | Episode 101 score = tensor([[2.]]), average score = 1.6732673645019531
# remaining: 15:44sec | [=====>...................] 20.00% | 200/1000 | started: 07:07:24 | Episode 201 score = tensor([[3.]]), average score = 2.1741292476654053
# remaining: 13:42sec | [=======>.................] 30.00% | 300/1000 | started: 07:07:24 | Episode 301 score = tensor([[0.]]), average score = 2.132890462875366
# remaining: 10:11sec | [==========>..............] 40.00% | 400/1000 | started: 07:07:24 | Episode 401 score = tensor([[1.]]), average score = 1.877805471420288
# remaining: 7:02sec | [============>............] 50.00% | 500/1000 | started: 07:07:24 | Episode 501 score = tensor([[1.]]), average score = 1.7405189275741577
# remaining: 5:06sec | [===============>.........] 60.00% | 600/1000 | started: 07:07:24 | Episode 601 score = tensor([[11.]]), average score = 1.80033278465271
# remaining: 3:44sec | [=================>.......] 70.00% | 700/1000 | started: 07:07:24 | Episode 701 score = tensor([[0.]]), average score = 1.8459343910217285
# remaining: 2:29sec | [====================>....] 80.00% | 800/1000 | started: 07:07:24 | Episode 801 score = tensor([[3.]]), average score = 1.911360740661621
# remaining: 1:17sec | [======================>..] 90.00% | 900/1000 | started: 07:07:24 | Episode 901 score = tensor([[2.]]), average score = 1.9145394563674927

def run(
        training_mode,
        pretrained,
        path="./models.ignore/dqn_chao_latent",
        number_of_episodes=1000,
        exploration_max=1,
        env_name="BreakoutNoFrameskip-v4",
        frame_buffer_size=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    ):
    
    env = TensorWrap(
        AutoLatentSpaceWrap(
                preprocess(
                env=gym.make(env_name),
                frame_buffer_size=frame_buffer_size,
                frame_sample_rate=frame_sample_rate,
            )
        )
    )
    
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        dropout=0.2,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
        pretrained=pretrained,
        path=path+"/",
        training_mode=training_mode,
    )
    
    mr_bond.when_mission_starts()
    for progress, episode_index in ProgressBar(range(number_of_episodes)):
        
        mr_bond.observation     = env.reset()
        mr_bond.reward          = torch.tensor([0]).unsqueeze(0)
        mr_bond.episode_is_over = torch.tensor([0]).unsqueeze(0)
        
        mr_bond.when_episode_starts(episode_index)
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            mr_bond.when_timestep_ends(timestep_index)
            
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    mr_bond.save()
    env.close()


run(training_mode=True, pretrained=False, number_of_episodes=1000, exploration_max=1)