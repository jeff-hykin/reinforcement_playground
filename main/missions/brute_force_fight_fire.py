import json_fix
import json
import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import time
from collections import defaultdict

from blissful_basics import flatten_once, product, max_index, to_pure, FS
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

def generate_random_memory_functions():
    while True:
        mapping = {}
        for each_possible_input in permutation_generator(input_vector_size, possible_values=[0,1]):
            mapping[tuple(each_possible_input)] = tuple(randomly_pick_from(
                tuple(permutation_generator(
                    memory_size,
                    possible_values=[0,1],
                ))
            ))
        random_memory_function = lambda observation: mapping[observation]
        random_memory_function.table = mapping
        yield random_memory_function
            

import json
from os.path import join
with open(FS.local_path('../world_builders/fight_fire/fire_fight_offline.ignore.json'), 'r') as in_file:
    timestep_json_list = json.load(in_file)

timesteps = [ Timestep.from_dict(each) for each in timestep_json_list ]
def evaluate_prediction_performance(memory_function):
    number_of_incorrect_predictions = 0
    reward_predictor_table = {}
    for each_timestep in timesteps:
        index        = each_timestep.index
        observation  = tuple(flatten(each_timestep.observation))
        response     = each_timestep.response
        reward       = each_timestep.reward
        is_last_step = each_timestep.is_last_step
        hidden_info  = each_timestep.hidden_info
        
        observation_and_memory = memory_function(observation)
        if observation_and_memory not in reward_predictor_table:
            reward_predictor_table[observation_and_memory] = reward
        else:
            predicted_value = reward_predictor_table[observation_and_memory]
            if predicted_value != reward:
                number_of_incorrect_predictions += 1
    
    return number_of_incorrect_predictions


def run_many_evaluations(iterations=100_000):
    memory_functions = []
    score_of = {}
    for progress, each_memory_function in ProgressBar(generate_random_memory_functions(), iterations=iterations):
        if progress.index >= iterations:
            break
        
        memory_functions.append(each_memory_function)
        number_incorrect = evaluate_prediction_performance(each_memory_function)
        score_of[each_memory_function] = -number_incorrect
        
        # logging and checkpoints
        if progress.updated:
            # 
            # update the terminal charts
            # 
            scores = tuple(score_of.values())
            progress.text = tui_distribution(create_buckets(scores, number_of_buckets=20))
            
            # 
            # save top 100  to disk
            # 
            sorted_memory_functions = sorted(memory_functions, key=lambda func: -score_of[func]) # python puts smallest values at the begining (so negative reverses that)
            top_100 = sorted_memory_functions[0:100]
            with_scores = [
                dict(
                    score=score_of[each_func],
                    table=each_func.table,
                )
                    for each_func in top_100
            ]
            FS.write(
                json.dumps(with_scores),
                to=FS.local_path("top_100_memory_maps.ignore.json")
            )
    
run_many_evaluations()