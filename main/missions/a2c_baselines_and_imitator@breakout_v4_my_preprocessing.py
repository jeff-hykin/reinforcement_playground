from time import time
import math
from collections import defaultdict
import functools

import torch
from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
from super_map import LazyDict
import math
from collections import defaultdict
import functools
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from gym.wrappers import AtariPreprocessing
from informative_iterator import ProgressBar
from statistics import mean as average


import tools.stat_tools as stat_tools
from tools.basics import product, flatten, to_pure
from tools.debug import debug
from tools.pytorch_tools import layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential
from tools.frame_que import FrameQue
from tools.schedulers import AgentLearningRateScheduler
from tools.agent_recorder import AgentRecorder

from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.helpful_fitness_measures import trend_up, average

from agent_builders.a2c_baselines.main import Agent as A2C
from agent_builders.auto_imitator.main import Agent # as Imitator
from world_builders.atari.custom_preprocessing import preprocess


def default_mission(
        env_name="BreakoutNoFrameskip-v4",
        number_of_episodes=500,
        frame_buffer_size=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        discount_factor=0.99,
        learning_rate=0.0007, # open ai defaults to 0.0007 for a2c
    ):
    
    env = preprocess(
        env=gym.make(env_name),
        frame_buffer_size=frame_buffer_size,
        frame_sample_rate=frame_sample_rate,
    )
    
    # mr_bond = A2C.load("models.ignore/BreakoutNoFrameskip-v4.zip")
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_0.00021598702086765554.model",
    )
    
    mr_bond.when_mission_starts()
    for progress, episode_index in ProgressBar(range(number_of_episodes)):
        print('episode_index = ', episode_index)
        if progress.updated and len(mr_bond.logging.episode_rewards) > 0:
            print("average reward: ", average(mr_bond.logging.episode_rewards))
        
        mr_bond.observation = env.reset()
        mr_bond.reward = 0
        mr_bond.episode_is_over = False
        
        mr_bond.when_episode_starts(episode_index)
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            print('mr_bond.action = ', mr_bond.action, 'timestep_index = ', timestep_index, 'mr_bond.episode_is_over = ', mr_bond.episode_is_over, 'mr_bond.reward = ', mr_bond.reward)
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