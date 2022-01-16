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
from tools.schedulers import AgentLearningRateScheduler

from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.helpful_fitness_measures import trend_up

from agent_builders.my_big_a2c.main import Agent


class PreprocessedBody:
    def __init__(self, brain, observation_space, action_space, frame_rate, frame_memory_size, frame_shape, **config):
        self.world_data = None # external world will change this
        self.action = None     # external world will read this
        # internal
        self.brain = brain
        self.observation_space = observation_space
        self.action_space = action_space
        self.frame_memory_size = frame_memory_size
        self.frame_rate = frame_rate
        self.frame_shape = frame_shape
        self.should_update_que = None
        self.observation = None
        self.episode_index = 0
        self.history = LazyDict(
            frames=[],
            rewards=[],
            episode_is_overs=[],
        )

    def when_mission_starts(self, *args):
        self.brain.when_mission_starts(*args)
    
    def when_episode_starts(self, episode_index):
        self.action      = self.action_space.sample() # take a random action before knowing what to do
        self.episode_index = episode_index
        self.brain.reward = 0
        self.brain.episode_is_over = False
        self.brain.observation = FrameQue(que_size=self.frame_memory_size, frame_shape=self.frame_shape)
        self.should_update_que   = Countdown(size=self.frame_rate)
        self.brain_timestep_occured = False
        self.first_brain_update = True
        self.should_update_brain = Countdown(
            size=self.frame_rate,
            offset=(self.frame_rate - 1), # because of possibly getting episode_is_over in the "in-between" frames
            delay=(self.frame_rate * (self.frame_memory_size - 1)), # wait to fill up the buffer
        )
    
    def when_timestep_starts(self, timestep_index):
        frame, _, _ = self.world_data
        self.history.frames.append(frame)
        
        if self.should_update_que():
            # FIXME: possibly add frame max-ing here
            self.brain.observation.add(frame)
        
        if self.should_update_brain():
            # edgecase of starting up the brain
            if self.first_brain_update:
                self.first_brain_update = False
                self.brain.when_episode_starts(self.episode_index)
            
            # self.brain.observation is already set
            self.brain.when_timestep_starts(timestep_index)
            # this is where delays or other desire-vs-reality changes would go
            self.action = self.brain.action
            self.brain_timestep_occured = True
    
    def when_timestep_ends(self, timestep_index):
        _, reward, episode_is_over = self.world_data
        self.history.rewards.append(reward)
        self.history.episode_is_overs.append(episode_is_over)
        
        if self.brain_timestep_occured:
            self.brain_timestep_occured = False
            # update brain with 2nd half of data
            self.brain.reward          = sum(self.history.rewards)
            self.brain.episode_is_over = any(self.history.episode_is_overs)
            # call it
            self.brain.when_timestep_ends(timestep_index)
            # reset the history
            self.history = LazyDict(
                frames=[],
                rewards=[],
                episode_is_overs=[],
            )
    
    def when_episode_ends(self, episode_index):
        self.brain.when_episode_ends(episode_index)
    
    def when_mission_ends(self, *args):
        self.brain.when_mission_ends(*args)
    

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
    
    env = gym.make(env_name)
    
    mr_bond = PreprocessedBody(
        brain=Agent(
            # observation_space is little bit hacky
            observation_space=LazyDict(shape=(frame_history, *env.observation_space.shape)),
            action_space=env.action_space,
            # live_updates=True,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
        ),
    )
    
    mr_bond.when_mission_starts()
    for episode_index in range(number_of_episodes):
        mr_bond.world_data = (env.reset(), 0, False)
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