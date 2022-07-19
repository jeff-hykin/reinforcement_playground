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

observation_size = 3 * 3 # three cells, three layers (self_position, fire_position, water_position)
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
            
        

perfect_agent_table = defaultdict(lambda *arguments: [True])
def perfect_agent(values):
    agent_position_1, agent_position_2, agent_position_3, *others = values
    rest_of_observation = others[:memory_size]
    memory = others[-memory_size:]
    if agent_position_1:
        memory_out = [ True ]
    else:
        memory_out = list(memory)
    key = tuple([ agent_position_1, agent_position_2, agent_position_3, *rest_of_observation, *memory ])
    perfect_agent_table[key] = memory_out
    return memory
    
perfect_agent.table = perfect_agent_table
    

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
        observation  = tuple(flatten(each_timestep.observation))
        response     = each_timestep.response
        reward       = each_timestep.reward
        is_last_step = each_timestep.is_last_step
        hidden_info  = each_timestep.hidden_info
        
        observation_and_memory = tuple(flatten((observation, memory_value)))
        if observation_and_memory not in reward_predictor_table:
            reward_predictor_table[observation_and_memory] = reward
        else:
            predicted_value = reward_predictor_table[observation_and_memory]
            if predicted_value != reward:
                number_of_incorrect_predictions += 1
        
        memory_value = memory_agent.get_next_memory_state(observation, memory_value)
    
    score = ( len(timesteps)-number_of_incorrect_predictions ) / len(timesteps)
    return score

def run_many_evaluations(iterations=10_000, competition_size=100):
    import math
    memory_agents = []
    next_generation = []
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