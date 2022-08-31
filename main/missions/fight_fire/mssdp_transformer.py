from collections import defaultdict
from copy import copy
import json
import pickle 
import random
import time

from blissful_basics import flatten_once, product, max_index, to_pure, flatten, print, FS, indent, singleton
from informative_iterator import ProgressBar
from super_hash import super_hash
from super_map import LazyDict
import ez_yaml
import gym
import json_fix
import math
import numpy as np
import torch

from agent_builders.dqn_lstm.universal_agent import Agent
from tools.universe.timestep import Timestep, TimestepSeries

def tuple_to_dict_hack_fix(data):
    return { f"{index}" : each for index, each in enumerate(data) }

def default_memory_reward_function(*, predicted_reward, real_reward):
    return -( (predicted_reward - real_reward)**2 )

class TransformedWorld:
    def __init__(transformed_world, *, memory_shape, real_env_factory, reward_predictor_factory, primary_agent_factory, memory_agent_factory):
        base_env = real_env_factory()
        transformed_world.space_for = LazyDict()
        transformed_world.space_for.memory             = gym.spaces.MultiBinary(product(memory_shape))
        transformed_world.space_for.wrapped_env        = gym.spaces.Dict( tuple_to_dict_hack_fix((transformed_world.space_for.memory, base_env.observation_space)) )
        transformed_world.space_for.reward_predictions = gym.spaces.Dict( tuple_to_dict_hack_fix((base_env.observation_space, base_env.action_space, transformed_world.space_for.memory)) )
        transformed_world.space_for.memory_env         = gym.spaces.Dict( tuple_to_dict_hack_fix((transformed_world.space_for.memory, base_env.observation_space, base_env.action_space)) )
        
        class RawRealEnvWithMemory:
            action_space      = base_env.action_space
            metadata          = base_env.metadata
            observation_space = transformed_world.space_for.wrapped_env
            def __init__(self, memory_source):
                self.real_env = real_env_factory()
                self.memory_source = memory_source
            
            def reset(self, *args):
                self.memory_source.memory_value = 0 # FIXME: probably need to make this a zeros_like(memory_shape) 
                observation = self.real_env.reset(*args)
                state = tuple_to_dict_hack_fix((self.memory_source.memory_value, observation))
                return state
            
            def step(self, action):
                next_observation, reward, done, info = self.real_env.step(action)
                next_state = tuple_to_dict_hack_fix((self.memory_source.memory_value, next_observation))
                return next_state, reward, done, info
        
        class MemoryEnv:
            metadata          = base_env.metadata
            action_space      = gym.spaces.MultiBinary(product(memory_shape))
            observation_space = transformed_world.space_for.memory_env
            
            def __init__(self, memory_reward_function=None):
                self.memory_reward_function = memory_reward_function or default_memory_reward_function
                self.memory_value = None
                self.real_env_with_memory = RawRealEnvWithMemory(memory_source=self)
                self.reward_predictor = reward_predictor_factory(
                    input_space=self.space_for.reward_predictions,
                )
                # init
                self.reset()
            
            def reset(self, *args):
                # 
                # seed the real env
                # 
                (self.memory_value, observation) = self.real_env_with_memory.reset(*args).values()
                self.primary_agent = primary_agent_factory(
                    observation_space=RawRealEnvWithMemory.observation_space,
                    action_space=RawRealEnvWithMemory.action_space,
                )
                # 
                # ..reset() main purpose == get first action
                # 
                primary_action = self.primary_agent.choose_action(
                    (self.memory_value, observation)
                )
                self.prev_observation = observation
                self.primary_agent_action = primary_action
                
                # 
                # put the inital memory_env state together
                # 
                memory_state = tuple_to_dict_hack_fix((self.memory_value, observation, primary_action))
                return memory_state
            
            def step(self, updated_memory_value):
                # 
                # get latest memory
                # 
                self.memory_value = updated_memory_value
                
                # 
                # get real_reward
                # 
                state, reward, done, info = self.real_env_with_memory.step(self.primary_agent_action)
                (_, current_observation) = state.values()
                
                # 
                # compute loss
                # 
                inputs=[
                    (self.prev_observation, self.primary_agent_action, updated_memory_value)
                ]
                predicted_reward = self.reward_predictor.predict(inputs)
                memory_reward_value = self.memory_reward_function(predicted_reward=predicted_reward, real_reward=reward)
                
                # 
                # train reward prediction
                # 
                self.reward_predictor.update(
                    inputs=inputs,
                    correct_outputs=[
                        reward
                    ]
                )
                
                # 
                # update/save variables
                # 
                self.prev_observation = current_observation
                self.primary_agent_action = self.primary_agent.choose_action(
                    tuple_to_dict_hack_fix((self.memory_value, current_observation))
                )
                memory_state = tuple_to_dict_hack_fix((self.memory_value, self.prev_observation, self.primary_agent_action))
                
                
                return memory_state, memory_reward_value, done, info
    
        class OriginalEnvWithMemory:
            action_space      = RawRealEnvWithMemory.action_space
            metadata          = RawRealEnvWithMemory.metadata
            observation_space = RawRealEnvWithMemory.observation_space
            
            def __init__(self):
                self.real_env = real_env_factory()
                self.memory_value = None
                self.memory_agent = memory_agent_factory(
                    action_space      = MemoryEnv.action_space,
                    observation_space = MemoryEnv.observation_space,
                )
            
            def reset(self, *args):
                self.memory_value = 0 # FIXME: probably need to make this a zeros_like(memory_shape) 
                self.prev_state = tuple_to_dict_hack_fix((self.memory_value, self.real_env.reset(*args)))
                return self.prev_state
            
            def step(self, action):
                # 
                # perform the action
                # 
                next_observation, reward, done, info = self.real_env.step(action)
                # 
                # compute latest memory value
                # 
                next_memory_env_state = (self.memory_value, next_observation, action)
                self.memory_value = self.memory_agent.choose_action(
                    next_memory_env_state
                )
                # 
                # let agent decide what to do next
                # 
                self.prev_state = tuple_to_dict_hack_fix((self.memory_value, next_observation))
                return self.prev_state, reward, done, info
        
        self.MemoryEnv             = MemoryEnv
        self.OriginalEnvWithMemory = OriginalEnvWithMemory
        pass 