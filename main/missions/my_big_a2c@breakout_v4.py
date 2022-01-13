from time import time
import math
from collections import defaultdict
import functools

import torch
from torch import nn
import gym
import numpy as np
# import silver_spectacle as ss
from super_map import LazyDict
import math
from collections import defaultdict
import functools
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from gym.wrappers import AtariPreprocessing

import tools.stat_tools as stat_tools
from tools.basics import product, flatten, to_pure
from tools.debug import debug
from tools.pytorch_tools import layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential
from tools.frame_que import FrameQue
from tools.schedulers import LearningRateScheduler

from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.helpful_fitness_measures import trend_up

from agent_builders.my_big_a2c.main import Agent

def default_mission(
        env_name="BreakoutNoFrameskip-v4",
        number_of_episodes=500,
        grayscale=True,
        frame_history=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        screen_size=84,
        discount_factor=0.99,
        learning_rate=0.0007, # open ai defaults to 0.0007 for a2c
    ):
    
    env = AtariPreprocessing(
        gym.make(env_name),
        grayscale_obs=grayscale,
        frame_skip=frame_sample_rate,
        noop_max=1, # no idea what this is, my best guess is; it is related to a do-dothing action and how many timesteps it does nothing for
        grayscale_newaxis=False, # keeps number of dimensions in observation the same for both grayscale and color (both have 4, b/c of the batch dimension)
    )
    
    mr_bond = Agent(
        # observation_space is little bit hacky
        observation_space=LazyDict(shape=(frame_history, *env.observation_space.shape)),
        action_space=env.action_space,
        # live_updates=True,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
    )
    
    mr_bond.when_mission_starts()
    for episode_index in range(number_of_episodes):
        mr_bond.episode_is_over = False
        # observation is a que of frames
        mr_bond.observation = FrameQue(que_size=frame_history, frame_shape=env.observation_space.shape) # fram que should probably be a wrapper around atari
        # add the latest frame
        mr_bond.observation.add(env.reset())
        mr_bond.when_episode_starts(episode_index)
        
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            latest_frame, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            # push in the newest frame
            mr_bond.observation.add(to_tensor(latest_frame))
            
            mr_bond.when_timestep_ends(timestep_index)
                
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    env.close()
    return mr_bond

def tune_hyperparams(number_of_episodes_per_trial=100_000, fitness_func=trend_up):
    import optuna
    # connect the trial-object to hyperparams and setup a measurement of fitness
    objective_func = lambda trial: fitness_func(
        default_mission(
            number_of_episodes=number_of_episodes_per_trial,
            discount_factor=trial.suggest_loguniform('discount_factor', 0.990, 0.991),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.00070, 0.00071),
        ).logging.episode_rewards
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=50)
    return study

# 
# run
# 
default_mission(
    number_of_episodes=100_000,
)