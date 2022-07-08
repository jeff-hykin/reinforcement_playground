import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import time
from collections import defaultdict

from blissful_basics import flatten_once, product, max_index, to_pure
from super_map import LazyDict
from super_hash import super_hash
from informative_iterator import ProgressBar

from world_builders.fight_fire.world import World
from agent_builders.dqn_lstm.universal_agent import Agent
from tools.universe.runtimes import basic
from tools.basics import sort_keys, randomly_pick_from, align

world = World(
    grid_width=3,
    grid_height=1,
    visualize=False,
    # debug=True,
)

# TODO: expectation validation, get a reward for having the correct prediction not for solving the environment
    # this is probably what is needed to overcome the greedy/probabilistic problem
    # may also be related to feature extraction

def run(number_of_timesteps_for_training=100_000, number_of_timesteps_for_testing=100_000):
    env = world.Player()
    mr_bond = Agent(
        observation_space=env.observation_space,
        response_space=env.action_space,
        responses=env.actions.values(),
    )
    
    # 
    # training
    # 
    for each_epsilon in [ 0.5, 0.05, 0 ]:
        mr_bond.epsilon = each_epsilon
        for progress, (episode_index, timestep) in ProgressBar(basic(agent=mr_bond, env=env), iterations=number_of_timesteps_for_training):
            world.random_seed = 1 # same world every time
            progress.text = f"""average_reward:{align(mr_bond.per_episode.average.reward, pad=4, decimals=0)}, reward: {align(mr_bond.episode.reward, digits=5, decimals=0)}, episode:{align(episode_index,pad=5)}, {align(mr_bond.epsilon*100, pad=3, decimals=0)}% random,
actions: {mr_bond.responses}
update value sum: {mr_bond._sum_table}
q value update-value: {mr_bond.q_value_per_decision}
ideal Q's: {mr_bond._ideal_table}
critic Q's: {mr_bond._critic_table}
policy decisions: {mr_bond.decision_table}"""
                
run()