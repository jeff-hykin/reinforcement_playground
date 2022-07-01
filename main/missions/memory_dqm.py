import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import time
from collections import defaultdict

from blissful_basics import flatten_once
from super_map import LazyDict

# from world_builders.frozen_lake.environment import Env
# from world_builders.cart_pole.environment import Env
from world_builders.fight_fire.world import World
from agent_builders.dqn_lstm.main import Agent
from tools.runtimes import traditional_runtime
from tools.basics import sort_keys, randomly_pick_from

from informative_iterator import ProgressBar

world = World(
    grid_width=3,
    grid_height=1,
    visualize=False,
    # debug=True,
)
Env = world.Player


# TODO: expectation validation, get a reward for having the correct prediction not for solving the environment
    # this is probably what is needed to overcome the greedy/probabilistic problem
    # may also be related to feature extraction



from blissful_basics import product, max_index, to_pure
from super_hash import super_hash
from tools.agent_skeleton import Skeleton

def align(value, pad=3, digits=5, decimals=3):
    # convert to int if needed
    if decimals == 0 and isinstance(value, float):
        value = int(value)
        
    if isinstance(value, int):
        return f"{value}".rjust(digits)
    elif isinstance(value, float):
        return f"{'{'}:{pad}.{decimals}f{'}'}".format(value).rjust(pad+decimals+1)
    else:
        return f"{value}".rjust(pad)

def run(number_of_timesteps_for_training=1_000, number_of_timesteps_for_testing=100_000):
    env = Env()
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        actions=env.actions.values(),
    )
    
    # 
    # training
    # 
    for each_epsilon in [ 0.9, 0.5, 0.05, 0 ]:
        print("# ")
        print(f"# {each_epsilon*100:.0f}% random")
        print("# ")
        mr_bond.epsilon = each_epsilon
        for progress, (episode_index, timestep_index, mr_bond.observation, action, mr_bond.reward, mr_bond.episode_is_over) in ProgressBar(traditional_runtime(agent=mr_bond, env=env), iterations=number_of_timesteps_for_training):
            world.random_seed = 1 # same world every time
            if mr_bond.episode_is_over:
                from statistics import mean as average
                progress.text = f"average_reward:{align(mr_bond.per_episode.average.reward, pad=4, decimals=0)}, reward: {align(mr_bond.episode.reward, digits=5, decimals=0)}, episode:{align(episode_index,pad=5)}, epsilon:{align(mr_bond.running_epsilon, pad=2, decimals=6)}, \n{mr_bond._table}"
            pass
    
    # 
    # testing
    # 
    mr_bond.training = False
    reward_sum = 0
    for progress, (episode_index, timestep_index, mr_bond.observation, action, mr_bond.reward, mr_bond.episode_is_over) in ProgressBar(traditional_runtime(agent=mr_bond, env=env), iterations=number_of_timesteps_for_testing):
        reward_sum += mr_bond.reward
        # print(progress)
        pass
        
    print(f'''reward_sum = {reward_sum}''')
    print(f'''reward_sum/episode_index = {reward_sum/(episode_index+1)}''')
    print(f'''reward_sum/timestep_index = {reward_sum/(timestep_index+1)}''')
    

run()