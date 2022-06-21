import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import time

# from world_builders.frozen_lake.environment import Env
# from world_builders.cart_pole.environment import Env
from world_builders.fight_fire.world import World
from agent_builders.dqn_primitive.main import Agent
from tools.runtimes import traditional_runtime

from informative_iterator import ProgressBar

world = World(grid_size=3)
Env = world.Player

def run(number_of_timesteps_for_training=10_000_000, number_of_timesteps_for_testing=1_000_000):
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
    reward_sum = 0
    for progress, (episode_index, timestep_index, mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over) in ProgressBar(traditional_runtime(agent=mr_bond, env=env), iterations=number_of_timesteps_for_training):
        reward_sum += max(reward_sum, mr_bond.reward)
        progress.text = f"reward: {reward_sum/(episode_index+1)}"
        pass
    
    # 
    # testing
    # 
    mr_bond.training = False
    reward_sum = 0
    for progress, (episode_index, timestep_index, mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over) in ProgressBar(traditional_runtime(agent=mr_bond, env=env), iterations=number_of_timesteps_for_testing):
        reward_sum += mr_bond.reward
        # print(progress)
        pass
        
    print(f'''reward_sum = {reward_sum}''')
    print(f'''reward_sum/episode_index = {reward_sum/(episode_index+1)}''')
    print(f'''reward_sum/timestep_index = {reward_sum/(timestep_index+1)}''')
    

run()