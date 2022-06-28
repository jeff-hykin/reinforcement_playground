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
from agent_builders.dqn_primitive.main import Agent
from tools.runtimes import traditional_runtime

from informative_iterator import ProgressBar

world = World(
    grid_size=3,
    # debug=True,
)
Env = world.Player


# TODO: expectation validation, get a reward for having the correct prediction not for solving the environment
    # this is probably what is needed to overcome the greedy/probabilistic problem
    # may also be related to feature extraction



from blissful_basics import product, max_index, to_pure
from super_hash import super_hash
from tools.agent_skeleton import Skeleton
class Agent(Skeleton):
    def __init__(self, 
        observation_space,
        action_space,
        actions=None,
        value_of=None,
        training=True,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.1,
        epsilon_decay=0.0000,
        default_value_assumption=0,
        bellman_update=None,
        get_best_action=None,
        memory_values=None,
    ):
        self.observation_space        = observation_space
        self.action_space             = action_space
        self.learning_rate            = learning_rate  
        self.discount_factor          = discount_factor
        self.epsilon                  = epsilon        # Amount of randomness in the action selection
        self.epsilon_decay            = epsilon_decay  # Fixed amount to decrease
        self.memory_values            = memory_values or [0,1] # boolean
        self.actions                  = tuple(flatten_once(    [ (each_action, each_memory_value) for each_action in actions ]     for each_memory_value in self.memory_values    ))
        self._table                   = defaultdict(lambda: self.default_value_assumption)
        self.value_of                 = value_of or (lambda state, action: self._table[to_pure((to_pure(state), to_pure(action)))])
        self.bellman_update           = bellman_update if callable(bellman_update) else (lambda state, action, value: self._table.update({ to_pure((to_pure(state), to_pure(action))): value }) )
        self.default_value_assumption = default_value_assumption
        self._get_best_action         = get_best_action
        self.training                 = training
        pass
    
    def when_mission_starts(self, mission_index=0):
        self.outcomes = []
        self.running_epsilon = self.epsilon if self.training else 0
        pass
        
    def when_episode_starts(self, episode_index):
        self.discounted_reward_sum = 0
        pass
        
    def when_timestep_starts(self, timestep_index):
        self.prev_observation = self.observation
        # if random number < epsilon, take a random action
        if self.action 
        if random.random() < self.running_epsilon:
            self.action = randomly_pick_from(self.actions)
        # else, take the action with the highest value in the current self.observation
        else:
            self.action = self.get_best_action(observation=self.observation)
        pass
    
    def get_best_action(self, observation):
        if isinstance(self.action_space, gym.spaces.Discrete):
            values = tuple((self.value_of(observation, each_action) for each_action in self.actions))
            best_action_key = max_index(values)
            return self.actions[best_action_key]
        elif callable(self._get_best_action):
            return self._get_best_action(self)
        else:
            raise Exception(f'''\n\nThe agent {self.__class__.__name__} doesn't have a way to choose the best action, please pass the argument: \n    {self.__class__.__name__}(get_best_action=lambda self: do_something(self.observation))\n\n''')
        pass
            
    def when_timestep_ends(self, timestep_index):
        old_q_value       = self.value_of(self.prev_observation, self.action)
        discounted_reward = self.reward + self.discount_factor * self.value_of(self.prev_observation, self.get_best_action(self.observation))
        self.discounted_reward_sum += discounted_reward
        
        # update q value
        new_value = old_q_value + self.learning_rate * (discounted_reward - self.value_of(self.prev_observation, self.action))
        self.bellman_update(self.prev_observation, self.action, new_value)
        pass
        
    def when_episode_ends(self, episode_index):
        self.outcomes.append(self.discounted_reward_sum)
        self.running_epsilon *= (1-self.epsilon_decay)
        pass
        
    def when_mission_ends(self, mission_index=0):
        pass

def randomly_pick_from(a_list):
    from random import randint
    index = randint(0, len(a_list)-1)
    return a_list[index]

def wrap(env):
    original_step = env.step
    original_reset = env.reset
    
    def new_step(action):
        action, memory = action
        state, reward, done, debug_info = original_step(action)
        return (state, memory), reward, done, debug_info
    
    def new_reset():
        return (original_reset(), 0)
    
    env.step = new_step
    env.reset = new_reset
    
    return env

def run(number_of_timesteps_for_training=10_000_000, number_of_timesteps_for_testing=1_000_000):
    env = Env()
    env = wrap(env)
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        actions=env.actions.values(),
    )
    
    # 
    # training
    # 
    reward_sum = 0
    action_freq = { each:0 for each in mr_bond.actions }
    for progress, (episode_index, timestep_index, mr_bond.observation, action, mr_bond.reward, mr_bond.episode_is_over) in ProgressBar(traditional_runtime(agent=mr_bond, env=env), iterations=number_of_timesteps_for_training):
        reward_sum += mr_bond.reward
        action_freq[action] += 1
        new = defaultdict(lambda : 0)
        new.update({ key: action_freq[key] for key in sorted(list(action_freq.keys())) })
        keys = list(mr_bond._table.keys())
        sorted_keys = sorted(keys)
        table       = { 
            key: mr_bond._table[key]
                for key in sorted_keys
        }
        progress.text = f"reward: {reward_sum/(episode_index+1)}, episode:{episode_index}, \n{dict(action_freq)}, epsilon:{mr_bond.running_epsilon}"
        # progress.text = f"reward: {reward_sum/(episode_index+1)}, episode:{episode_index}, \n{dict(action_freq)}, \n{LazyDict(table)}"
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