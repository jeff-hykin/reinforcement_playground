from collections import defaultdict
from copy import copy
import json
import pickle 
import random
import time

from blissful_basics import flatten_once, product, max_index, to_pure, flatten, print, FS
from informative_iterator import ProgressBar
from super_hash import super_hash
from super_map import LazyDict
import ez_yaml
import gym
import json_fix
import math
import numpy as np
import torch
import torch.nn as nn

from world_builders.fight_fire.world import World
from agent_builders.dqn_lstm.universal_agent import Agent
from tools.universe.timestep import Timestep
from tools.universe.runtimes import basic
from tools.basics import project_folder, sort_keys, randomly_pick_from, align, create_buckets, tui_distribution, permutation_generator

number_of_timesteps = 1200
corridor_length   = 7
action_length     = 2
memory_size       = 1
observation_size  = corridor_length + action_length
input_vector_size = observation_size + memory_size
verbose           = False

# 
# memory agent
# 
if True:
    _number_of_memory_agents = 0
    class MemoryAgent:
        def __init__(self, id=None, is_stupid=False, table=None):
            # 
            # set id
            # 
            global _number_of_memory_agents
            if type(id) != type(None):
                self.id = id
            else:
                _number_of_memory_agents += 1
                self.id = _number_of_memory_agents
            
            # 
            # is_stupid
            # 
            self.is_stupid = is_stupid
            
            # 
            # table
            # 
            if table:
                self.table = table
            else:
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
            
            if self.is_stupid:
                return [ False for each in range(memory_size) ]
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
            return MemoryAgent(
                is_stupid=self.is_stupid,
                table=copy(self.table),
            )

        def generate_mutated_copy(self, number_of_mutations):
            duplicate = self.duplicate()
            keys = list(duplicate.table.keys())
            random.shuffle(keys)
            selected_keys = keys[0:number_of_mutations]
            
            for each in selected_keys:
                duplicate.table[each] = MemoryAgent.random_memory_configuration()
            
            return duplicate
        
        def __json__(self):
            json_friendly_table = [
                [ each_key, each_value ]
                    for each_key, each_value in self.table.items()
            ]
            return {
                "class": "MemoryAgent",
                "kwargs": dict(
                    id=self.id,
                    is_stupid=self.is_stupid
                ),
                "json_friendly_table": json_friendly_table,
            }
        
        @staticmethod
        def from_json(json_data):
            # parse if needed
            if isinstance(json_data, str):
                import json
                json_data = json.loads(json_data)
            
            real_table = {}
            for each_key, each_value in json_data["json_friendly_table"]:
                real_table[each_key] = each_value
            
            json_data["kwargs"]["table"] = real_table
            return MemoryAgent(**json_data["kwargs"])

    class PerfectMemoryAgent(MemoryAgent):
        def __init__(self):
            self.table = {}
            self.id = 0
        
        def get_next_memory_state(self, observation, prev_memory):
            agent_position_0, *others = observation
            key = tuple(flatten([observation, prev_memory])) if type(prev_memory) != type(None) else tuple(observation)
            
            prev_memory = flatten(prev_memory)
            if agent_position_0:
                memory_out = [ True ]
            else:
                memory_out = list(prev_memory)
                
            self.table[key] = [ not not each for each in memory_out ]
            return self.table[key]
            
        def duplicate(self):
            return self
        
        def generate_mutated_copy(self, number_of_mutations):
            a_copy = MemoryAgent()
            a_copy.table.update(self.table)
            return a_copy

# 
# evaluation function
# 
if True:
    # max_number_of_eval_timesteps = 100
    # import json
    # from os.path import join
    # with open(FS.local_path(f'{project_folder}/main/world_builders/fight_fire/fire_fight_offline.ignore.json'), 'r') as in_file:
    #     timestep_json_list = json.load(in_file)
    
    # timesteps = list(enumerate([
    #     Timestep(**each)
    #         for each in timestep_json_list[0:max_number_of_eval_timesteps]
    # ]))
    
    # timesteps += [
    #     (1 , Timestep(index=0, observation=[[False,True ,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=1))),
    #     (2 , Timestep(index=1, observation=[[False,False,True ]], response="RIGHT", reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=1))),
    #     (3 , Timestep(index=2, observation=[[False,False,True ]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=1))),
    #     (4 , Timestep(index=3, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=1))),
    #     (5 , Timestep(index=4, observation=[[True ,False,False]], response="LEFT" , reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=1))),
    #     (6 , Timestep(index=5, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=1))),
    #     (7 , Timestep(index=6, observation=[[False,True ,False]], response="RIGHT", reward=0.05  , is_last_step=True , hidden_info=LazyDict(episode_index=1))),
    
    #     (8 , Timestep(index=0, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=2))),
    #     (9 , Timestep(index=1, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=2))),
    #     (10, Timestep(index=2, observation=[[False,True ,False]], response="RIGHT", reward=0.05  , is_last_step=True , hidden_info=LazyDict(episode_index=2))),
    
    #     (11, Timestep(index=0, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=3))),
    #     (12, Timestep(index=1, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=3))),
    #     (13, Timestep(index=2, observation=[[False,True ,False]], response="RIGHT", reward=0.05  , is_last_step=True , hidden_info=LazyDict(episode_index=3))),
    
    #     (14, Timestep(index=0, observation=[[False,True ,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (16, Timestep(index=1, observation=[[False,False,True ]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (14, Timestep(index=2, observation=[[False,True ,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (16, Timestep(index=3, observation=[[False,False,True ]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (17, Timestep(index=4, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (18, Timestep(index=5, observation=[[True ,False,False]], response="LEFT" , reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (19, Timestep(index=6, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (17, Timestep(index=7, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (19, Timestep(index=8, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=4))),
    #     (20, Timestep(index=9, observation=[[False,True ,False]], response="RIGHT", reward=0.05  , is_last_step=True , hidden_info=LazyDict(episode_index=4))),
    
    #     (21, Timestep(index=0, observation=[[False,True ,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=5))),
    #     (22, Timestep(index=1, observation=[[False,False,True ]], response="RIGHT", reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=5))),
    #     (23, Timestep(index=2, observation=[[False,False,True ]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=5))),
    #     (24, Timestep(index=3, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=5))),
    #     (25, Timestep(index=4, observation=[[True ,False,False]], response="LEFT" , reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=5))),
    #     (26, Timestep(index=5, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=5))),
    #     (27, Timestep(index=6, observation=[[False,True ,False]], response="RIGHT", reward=0.05  , is_last_step=True , hidden_info=LazyDict(episode_index=5))),
    
    #     (28, Timestep(index=0, observation=[[False,True ,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=6))),
    #     (29, Timestep(index=1, observation=[[False,False,True ]], response="RIGHT", reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=6))),
    #     (30, Timestep(index=2, observation=[[False,False,True ]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=6))),
    #     (31, Timestep(index=3, observation=[[False,True ,False]], response="LEFT" , reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=6))),
    #     (32, Timestep(index=4, observation=[[True ,False,False]], response="LEFT" , reward=-0.02 , is_last_step=False, hidden_info=LazyDict(episode_index=6))),
    #     (33, Timestep(index=5, observation=[[True ,False,False]], response="RIGHT", reward=-0.001, is_last_step=False, hidden_info=LazyDict(episode_index=6))),
    #     (34, Timestep(index=6, observation=[[False,True ,False]], response="RIGHT", reward=0.05  , is_last_step=True , hidden_info=LazyDict(episode_index=6))),
    # ]
    timesteps = []

    @print.indent.function_block
    def evaluate_prediction_performance(memory_agent):
        global timesteps
        if not timesteps:
            timesteps = list(enumerate(generate_samples(number_of_timesteps=number_of_timesteps)))
        
        is_perfect_agent = isinstance(memory_agent, PerfectMemoryAgent)
        print(f"memory_agent: {memory_agent.id}")
        print(f'''is_perfect_agent = {is_perfect_agent}''')
        
        number_of_incorrect_predictions = 0
        reward_predictor_table = {}
        memory_value = None
        was_last_step = True
        for training_index, each_timestep in timesteps:
            if was_last_step: memory_value = None
            index         = each_timestep.index
            observation   = each_timestep.observation
            response      = each_timestep.response
            reward        = each_timestep.reward
            is_last_step  = each_timestep.is_last_step
            hidden_info   = each_timestep.hidden_info
            episode_index = each_timestep.hidden_info["episode_index"]
            
            state_input = simplify_observation_and_reaction(observation, response)
            
            with print.indent.block(f"episode: {episode_index}"):
                memory_value = memory_agent.get_next_memory_state(state_input, memory_value)
                observation_and_memory = tuple(flatten((state_input, memory_value)))
                if observation_and_memory not in reward_predictor_table:
                    reward_predictor_table[observation_and_memory] = reward
                    with print.indent.block(f"{training_index}: reward init"):
                        print(f'''observation_and_memory = {observation_and_memory_as_human_string(observation_and_memory)}''')
                        print(f'''reward = {reward}''')
                else:
                    predicted_value = reward_predictor_table[observation_and_memory]
                    was_wrong = predicted_value != reward
                    if was_wrong:
                        number_of_incorrect_predictions += 1
                    with print.indent.block(f"{training_index}: reward check"):
                        print(f'''observation_and_memory = {observation_and_memory_as_human_string(observation_and_memory)}''')
                        print(f'''reward = {reward}''')
                        print(f'''predicted_value: {predicted_value}''')
                        print(f'''was_wrong: {was_wrong}''')
                
            was_last_step = is_last_step
        
        score = ( len(timesteps)-number_of_incorrect_predictions ) / len(timesteps)
        print(f'''score = {score}''')
        return score

# 
# runtime
# 
def run_many_evaluations(iterations=3, competition_size=100, genetic_method="mutation", disable_memory=False, deviation_proportion=0.1, enable_perfect=False):
    import math
    memory_agents = []
    next_generation = [ PerfectMemoryAgent() ] if enable_perfect else []
    score_of = {}
    
    for each in range(competition_size):
        next_generation.append(MemoryAgent(is_stupid=disable_memory))
    
    # 
    # for updating terminal charts
    # 
    def compute_terminal_chart():
        scores = tuple(score_of.values())
        max_score = max(scores)
        buckets, bucket_ranges = create_buckets(scores, number_of_buckets=20)
        buckets, bucket_ranges = reversed(buckets), reversed(bucket_ranges)
        a_string = tui_distribution(buckets, [ f"( {small*100:3.2f}, {big*100:3.2f} ]" for small, big in bucket_ranges ])
        return a_string
    
    for progress, *_ in ProgressBar(iterations):
        if progress.index >= iterations:
            break
        
        # evaluate new ones
        with print.indent:
            for each_memory_agent in next_generation:
                if not verbose: print.disable.always = True
                score_of[id(each_memory_agent)] = evaluate_prediction_performance(each_memory_agent)
                if not verbose: print.disable.always = False
        # only keep top 100
        with print.indent:
            memory_agents += next_generation
            sorted_memory_agents = sorted(memory_agents, key=lambda func: -score_of[id(func)]) # python puts smallest values at the begining (so negative reverses that)
            top_100 = sorted_memory_agents[0:competition_size]
            memory_agents = top_100
        # create next generation
        with print.indent:
            next_generation.clear()
            number_of_values = len(top_100[0].table.values())
            for each_memory_agent in memory_agents:
                if genetic_method == "mutation":
                    number_of_mutations = math.floor(random.random()/deviation_proportion * number_of_values) # random % of all values
                    next_generation.append(
                        each_memory_agent.generate_mutated_copy(number_of_mutations)
                    )
                else:
                    next_generation.append(
                        MemoryAgent(is_stupid=disable_memory)
                    )
        # logging and checkpoints
        if progress.updated:
            progress.pretext += compute_terminal_chart()
            
            # 
            # save top 10  to disk
            # 
                # sorted_memory_agents = sorted(memory_agents, key=lambda func: -score_of[id(func)]) # python puts smallest values at the begining (so negative reverses that)
                # top_10 = sorted_memory_agents[0:10]
                # with_scores = [
                #     dict(
                #         score=score_of[id(each_func)],
                #         table=each_func.table,
                #     )
                #         for each_func in top_10
                # ]
                # path = FS.local_path("top_10_memory_maps.ignore.yaml")
                # FS.write(
                #     ez_yaml.to_string(obj=with_scores),
                #     to=path,
                # )
    print(compute_terminal_chart())

# 
# sample generator
# 
def generate_samples(number_of_timesteps=200):
    offline_timesteps = []
    
    world = World(
        grid_width=corridor_length,
        grid_height=1,
        visualize=False,
        # debug=True,
        corridor_mode=True,
    )
    env = world.Player()
    mr_bond = Agent(
        observation_space=env.observation_space,
        response_space=env.action_space,
        responses=env.actions.values(),
    )
    
    # 
    # training
    # 
    for each_epsilon in [ 1.0 ]:
        mr_bond.epsilon = each_epsilon
        for progress, (episode_index, timestep) in ProgressBar(basic(agent=mr_bond, env=env), iterations=number_of_timesteps):
            timestep.hidden_info = dict(episode_index=episode_index)
            offline_timesteps.append(timestep)
            world.random_seed = 1 # same world every time
            progress.text = f"""average_reward:{align(mr_bond.per_episode.average.reward, pad=4, decimals=0)}, reward: {align(mr_bond.episode.reward, digits=5, decimals=0)}, episode:{align(episode_index,pad=5)}, {align(mr_bond.epsilon*100, pad=3, decimals=0)}% random,\n{mr_bond.debug}"""
        
    return offline_timesteps


# 
# helpers
# 
if True:
    def observation_and_memory_as_human_string(observation_and_memory):
        keys = [ *([f"position{each}" for each in range(corridor_length)]), "going_left", "going_right", "memory"]
        with_names = [ f"{each_key}:{(each_val or f'{each_val}'.lower())}" for each_key, each_val in zip(keys, observation_and_memory)]
        return ", ".join(with_names)

    def simplify_observation_and_reaction(observation, action):
        observation = list(flatten(to_pure(observation)))
        observation = observation[0:corridor_length]
        if action == "UP":
            action = [ True , True  ]
        elif action == "DOWN":
            action = [ False, False ]
        elif action == "LEFT":
            action = [ True , False ]
        elif action == "RIGHT":
            action = [ False, True ]
        else:
            raise Exception(f'''
                action = {action}
                expected up/down/left/right
            ''')
        
        return tuple(flatten(observation + action))

print.flush.always = False # optimization    

print("#")
print("# genetic_mutations_with_perfect")
print("#")
with print.indent:
    run_many_evaluations(iterations=5, genetic_method="mutation", disable_memory=False, enable_perfect=True)

print("#")
print("# genetic_mutations")
print("#")
with print.indent:
    run_many_evaluations(iterations=25*4, genetic_method="mutation", disable_memory=False)

print("#")
print("# pure random")
print("#")
with print.indent:
    run_many_evaluations(iterations=5, genetic_method="random", disable_memory=False)

print("#")
print("# no_memory")
print("#")
with print.indent:
    run_many_evaluations(iterations=5, genetic_method="random", disable_memory=True)