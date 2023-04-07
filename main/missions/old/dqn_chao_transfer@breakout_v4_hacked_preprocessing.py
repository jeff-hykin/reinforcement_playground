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

# 22 min: Episode 1000 score = 9.0, average score = 5.572

from world_builders.atari.custom_preprocessing import preprocess
from world_builders.atari.preprocessor_chao import TensorWrap
from agent_builders.dqn_chao_transfer.main import Agent

from prefabs.auto_imitator.main import AutoImitator
from informative_iterator import ProgressBar
from tools.pytorch_tools import to_tensor

# remaining: 1:03:23sec | [==>......................] 10.00% | 100/1000 | started: 07:19:29 | Episode 101 score = tensor([[10.]]), average score = 3.267326831817627
# remaining: 50:31sec | [=====>...................] 20.00% | 200/1000 | started: 07:19:29 | Episode 201 score = tensor([[2.]]), average score = 3.631840705871582
# remaining: 43:17sec | [=======>.................] 30.00% | 300/1000 | started: 07:19:29 | Episode 301 score = tensor([[2.]]), average score = 4.016611099243164
# remaining: 36:38sec | [==========>..............] 40.00% | 400/1000 | started: 07:19:29 | Episode 401 score = tensor([[7.]]), average score = 4.177057266235352
# remaining: 30:24sec | [============>............] 50.00% | 500/1000 | started: 07:19:29 | Episode 501 score = tensor([[8.]]), average score = 4.403193473815918
# remaining: 24:18sec | [===============>.........] 60.00% | 600/1000 | started: 07:19:29 | Episode 601 score = tensor([[11.]]), average score = 4.60565710067749
# remaining: 18:38sec | [=================>.......] 70.00% | 700/1000 | started: 07:19:29 | Episode 701 score = tensor([[7.]]), average score = 4.634807586669922
# remaining: 12:44sec | [====================>....] 80.00% | 800/1000 | started: 07:19:29 | Episode 801 score = tensor([[0.]]), average score = 4.700374603271484
# remaining: 6:22sec | [======================>..] 90.00% | 900/1000 | started: 07:19:29 | Episode 901 score = tensor([[4.]]), average score = 4.630410671234131

def run(
        training_mode,
        pretrained,
        path="./models.ignore/dqn_chao_transfer",
        number_of_episodes=1000,
        exploration_max=1,
        env_name="BreakoutNoFrameskip-v4",
        frame_buffer_size=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    ):
    
    auto_imitator = AutoImitator(
        learning_rate=0.00021,
        input_shape=(4,84,84),
        latent_shape=(512,),
        output_shape=(4,),
        path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_0.00021598702086765554.model",
    )
    
    env = TensorWrap(
        preprocess(
            env=gym.make(env_name),
            frame_buffer_size=frame_buffer_size,
            frame_sample_rate=frame_sample_rate,
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
        encoder=auto_imitator.encoder,
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