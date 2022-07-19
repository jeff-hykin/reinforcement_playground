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
from tools.basics import sort_keys, randomly_pick_from, align

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
        yield lambda observation: mapping[observation]
            

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


def run_many_evaluations():
    pass
        