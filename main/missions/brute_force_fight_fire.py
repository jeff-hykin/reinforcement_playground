import json_fix
import json
import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import time
import math
from copy import copy
from collections import defaultdict

import ez_yaml
from blissful_basics import flatten_once, product, max_index, to_pure, flatten, FS
from super_map import LazyDict
from super_hash import super_hash
from informative_iterator import ProgressBar

from world_builders.fight_fire.world import World
from agent_builders.dqn_lstm.universal_agent import Agent
from tools.universe.timestep import Timestep
from tools.universe.runtimes import basic
from tools.basics import sort_keys, randomly_pick_from, align, create_buckets, tui_distribution

observation_size = 5 # position (3), action (2)
memory_size = 1
input_vector_size = observation_size + memory_size

def permutation_generator(digits, possible_values):
    if digits == 1:
        for each in possible_values:
            yield [ each ]
    elif digits > 1:
        for each_subcell in permutation_generator(digits-1, possible_values):
            for each in possible_values:
                yield [ each ] + each_subcell
    # else: dont yield anything

def convert_action_observation(observation, action):
    observation = observation[0:3]
    if action == "UP":
        action = [ 1, 1 ]
    if action == "DOWN":
        action = [ 0, 0 ]
    if action == "LEFT":
        action = [ 1, 0 ]
    if action == "RIGHT":
        action = [ 0, 1 ]
    return tuple(flatten(observation + action))

class MemoryAgent:
    def __init__(self, function_helper=None):
        self.function_helper = function_helper
        mapping = {}
        # start values (no memory)
        for each_possible_input in permutation_generator(input_vector_size-1, possible_values=[True,False]):
            mapping[tuple(each_possible_input)] = MemoryAgent.random_memory_configuration()
        # subsequent values (memory as input)
        for each_possible_input in permutation_generator(input_vector_size, possible_values=[True,False]):
            mapping[tuple(each_possible_input)] = MemoryAgent.random_memory_configuration()
        
        self.table = mapping
    
    def get_next_memory_state(self, observation, memory_value):
        if memory_value is None:
            key = observation
        else:
            key = tuple(flatten((observation, memory_value)))
        
        if key not in self.table:
            if callable(self.function_helper):
                output = self.function_helper(observation, memory_value)
                self.table[key] = output
                return output
        else:
            return self.table[key]
        
    
    @staticmethod
    def random_memory_configuration():
        return tuple(randomly_pick_from(
            tuple(permutation_generator(
                memory_size,
                possible_values=[True,False],
            ))
        ))
    
    def duplicate(self):
        the_copy = MemoryAgent(self.function_helper)
        the_copy.table = copy(self.table)
        return the_copy

    def generate_mutated_copy(self, number_of_mutations):
        duplicate = self.duplicate()
        keys = list(duplicate.table.keys())
        random.shuffle(keys)
        selected_keys = keys[0:number_of_mutations]
        
        for each in selected_keys:
            duplicate.table[each] = MemoryAgent.random_memory_configuration()
        
        return duplicate

class PerfectMemoryAgent(MemoryAgent):
    def __init__(self):
        self.table = {}
    
    def get_next_memory_state(self, observation, memory_value):
        agent_position_0, *others = observation
        key = tuple(flatten([observation, memory_value])) if type(memory_value) != type(None) else tuple(observation)
        
        memory_value = flatten(memory_value)
        if agent_position_0:
            memory_out = [ True ]
        else:
            memory_out = list(memory_value)
            
        self.table[key] = memory_out
        return memory_out
        
    def duplicate(self):
        return self
    
    def generate_mutated_copy(self, number_of_mutations):
        a_copy = MemoryAgent()
        a_copy.table.update(self.table)
        return a_copy


import json
from os.path import join
with open(FS.local_path('../world_builders/fight_fire/fire_fight_offline.ignore.json'), 'r') as in_file:
    timestep_json_list = json.load(in_file)

max_number_of_eval_timesteps = 1000
timesteps = [ Timestep.from_dict(each) for each in timestep_json_list ][:max_number_of_eval_timesteps]
def evaluate_prediction_performance(memory_agent):
    number_of_incorrect_predictions = 0
    reward_predictor_table = {}
    memory_value = None
    for each_timestep in timesteps:
        index        = each_timestep.index
        observation  = each_timestep.observation
        response     = each_timestep.response
        reward       = each_timestep.reward
        is_last_step = each_timestep.is_last_step
        hidden_info  = each_timestep.hidden_info
        
        state_input = convert_action_observation(observation, response)
        
        observation_and_memory = tuple(flatten((state_input, memory_value)))
        if observation_and_memory not in reward_predictor_table:
            reward_predictor_table[observation_and_memory] = reward
        else:
            predicted_value = reward_predictor_table[observation_and_memory]
            if predicted_value != reward:
                number_of_incorrect_predictions += 1
        
        memory_value = memory_agent.get_next_memory_state(state_input, memory_value)
    
    score = ( len(timesteps)-number_of_incorrect_predictions ) / len(timesteps)
    return score

def run_many_evaluations(iterations=10_000, competition_size=100):
    import math
    memory_agents = []
    next_generation = [
        PerfectMemoryAgent()
    ]
    score_of = {}
    
    for each in range(competition_size):
        next_generation.append(MemoryAgent())
    
    for progress, *_ in ProgressBar(iterations):
        if progress.index >= iterations:
            break
        
        # evaluate new ones
        for each_memory_agent in next_generation:
            score_of[id(each_memory_agent)] = evaluate_prediction_performance(each_memory_agent)
        memory_agents += next_generation
        sorted_memory_agents = sorted(memory_agents, key=lambda func: -score_of[id(func)]) # python puts smallest values at the begining (so negative reverses that)
        top_100 = sorted_memory_agents[0:100]
        memory_agents = top_100
        number_of_mutations = math.floor(random.random() * len(top_100[0].table.values())) # ranomd % of all values
        next_generation.clear()
        for each_memory_agent in memory_agents:
            next_generation.append(
                each_memory_agent.generate_mutated_copy(number_of_mutations)
            )
        
        # logging and checkpoints
        if progress.updated:
            # 
            # update the terminal charts
            # 
            scores = tuple(score_of.values())
            max_score = max(scores)
            buckets, bucket_ranges = create_buckets(scores, number_of_buckets=20)
            buckets, bucket_ranges = reversed(buckets), reversed(bucket_ranges)
            progress.pretext += tui_distribution(buckets, [ f"( {small*100:3.2f}, {big*100:3.2f} ]" for small, big in bucket_ranges ])
            
            # 
            # save top 100  to disk
            # 
            sorted_memory_agents = sorted(memory_agents, key=lambda func: -score_of[id(func)]) # python puts smallest values at the begining (so negative reverses that)
            top_10 = sorted_memory_agents[0:10]
            with_scores = [
                dict(
                    score=score_of[id(each_func)],
                    table=each_func.table,
                )
                    for each_func in top_10
            ]
            path = FS.local_path("top_100_memory_maps.ignore.yaml")
            FS.write(
                ez_yaml.to_string(obj=with_scores),
                to=path,
            )
    
run_many_evaluations()