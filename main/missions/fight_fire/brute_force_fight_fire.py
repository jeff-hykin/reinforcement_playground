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

number_of_timesteps = 500
corridor_length   = 17
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
        def get_next_memory_state(self, observation, memory_value):
            return [ False for each in range(memory_size) ]
    
    class MemoryTriggerAgent:
        def __init__(self, id=None, is_stupid=False, triggers=None, number_of_triggers=1):
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
            if triggers:
                self.triggers = triggers
            else:
                self.triggers = [ self.generate_random_trigger() for each in range(number_of_triggers) ]
        
        def generate_random_trigger(self):
            """
            returns input_trigger_conditions={ input_index : required_value }, new_memory_mapping={ memory_index: new output value }
            """
            # 
            # what to pay attention to
            # 
            how_many_values_to_pay_attention_to = random.randint(1,input_vector_size)
            indicies_of_inputs = [ each for each in range(input_vector_size) ]
            random.shuffle(indicies_of_inputs)
            selected_indicies = indicies_of_inputs[0:how_many_values_to_pay_attention_to]
            input_trigger_conditions = {
                input_index : randomly_pick_from([ True, False ])
                    for input_index in selected_indicies 
            }
            
            # 
            # where/what to output
            # 
            how_many_memory_values_to_set = random.randint(1,input_vector_size)
            indicies_of_memory = [ each for each in range(memory_size) ]
            random.shuffle(indicies_of_memory)
            selected_memory_indicies = indicies_of_memory[0:how_many_memory_values_to_set]
            new_memory_mapping = {
                memory_index : randomly_pick_from([ True, False ])
                    for memory_index in selected_memory_indicies 
            }
            
            return input_trigger_conditions, new_memory_mapping
    
        def get_next_memory_state(self, observation, memory_value):
            if self.is_stupid:
                return [False] * memory_size
            
            input_vector = flatten([ observation, memory_value ])
            for input_trigger_conditions, new_memory_mapping in self.triggers:
                failed_conditions = False
                for each_key, each_value in input_trigger_conditions.items():
                    if input_vector[each_key] != each_value:
                        failed_conditions = True
                        break
                if failed_conditions: continue
                        
                # if all the checks pass
                memory_copy = list(flatten(memory_value))
                for each_key, each_value in new_memory_mapping.items():
                    memory_copy[each_key] = each_value
                
                return memory_copy 
            
            # one technically hardcoded trigger
            if type(memory_value) == type(None):
                return [False] * memory_size
            
            # if all triggers fail, preserve memory
            return tuple(memory_value)
            
        def generate_mutated_copy(self, number_of_mutations):
            # just fully randomize it since its hard to mutate
            # TODO: change this in the future
            return MemoryTriggerAgent()
        
        def __json__(self):
            return {
                "class": "MemoryTriggerAgent",
                "kwargs": dict(
                    id=self.id,
                    is_stupid=self.is_stupid,
                    triggers=self.triggers,
                ),
            }
        
        @staticmethod
        def from_json(json_data):
            return MemoryTriggerAgent(**json_data["kwargs"])

    
    class MemoryMapAgent(MemoryAgent):
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
                    mapping[tuple(each_possible_input)] = MemoryMapAgent.random_memory_configuration()
                # subsequent values (memory as input)
                for each_possible_input in permutation_generator(input_vector_size, possible_values=[True,False]):
                    mapping[tuple(each_possible_input)] = MemoryMapAgent.random_memory_configuration()
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
            return MemoryMapAgent(
                is_stupid=self.is_stupid,
                table=copy(self.table),
            )

        def generate_mutated_copy(self, number_of_mutations):
            duplicate = self.duplicate()
            keys = list(duplicate.table.keys())
            random.shuffle(keys)
            selected_keys = keys[0:number_of_mutations]
            
            for each in selected_keys:
                duplicate.table[each] = MemoryMapAgent.random_memory_configuration()
            
            return duplicate
        
        def __json__(self):
            json_friendly_table = [
                [ each_key, each_value ]
                    for each_key, each_value in self.table.items()
            ]
            return {
                "class": "MemoryMapAgent",
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
            return MemoryMapAgent(**json_data["kwargs"])

    class PerfectMemoryMapAgent(MemoryAgent):
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
    
    class PerfectMemoryTriggerAgent(MemoryTriggerAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.id = 0
            self.triggers = [
                [
                    # conditions (position0 needs to be True)
                    { 0 : True }, 
                    # consequences (memory slot 0 is set to true)
                    { 0 : True }
                ]
            ]
        
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
    timesteps = []
    @print.indent.function_block
    def evaluate_prediction_performance(memory_agent):
        global timesteps
        if not timesteps:
            timesteps = list(enumerate(generate_samples(number_of_timesteps=number_of_timesteps)))
        
        is_perfect_agent = isinstance(memory_agent, PerfectMemoryMapAgent)
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
    next_generation = [ PerfectMemoryTriggerAgent() ] if enable_perfect else []
    score_of = {}
    
    for each in range(competition_size):
        next_generation.append(MemoryTriggerAgent(is_stupid=disable_memory))
    
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
            for each_memory_agent in memory_agents:
                if genetic_method == "mutation":
                    number_of_values = len(top_100[0].table.values())
                    number_of_mutations = math.floor(random.random()/deviation_proportion * number_of_values) # random % of all values
                    next_generation.append(
                        each_memory_agent.generate_mutated_copy(number_of_mutations)
                    )
                else:
                    next_generation.append(
                        MemoryTriggerAgent(is_stupid=disable_memory)
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
    
    # return the best score
    return max(score_of.values())

# 
# sample generator
# 
def generate_samples(number_of_timesteps):
    offline_timesteps = []
    
    world = World(
        grid_width=corridor_length,
        grid_height=1,
        visualize=False,
        # debug=True,
        corridor_mode=True,
    )
    env = world.Player()
    
    import torch
    import torch.nn as nn
    import random
    from tqdm import tqdm
    import pickle 
    import gym
    import numpy as np
    import collections 
    import cv2
    import time
    from collections import defaultdict
    from copy import deepcopy

    from blissful_basics import product, max_index, flatten
    from super_hash import super_hash
    from super_map import LazyDict

    from tools.universe.agent import Skeleton, Enhancement, enhance_with
    from tools.universe.timestep import Timestep
    from tools.universe.enhancements.basic import EpisodeEnhancement, LoggerEnhancement
    from tools.file_system_tools import FileSystem
    from tools.stat_tools import normalize

    from tools.debug import debug
    from tools.basics import sort_keys, randomly_pick_from
    from tools.object import Object, Options

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from prefabs.simple_lstm import SimpleLstm
    from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier, all_argmax_coordinates
    from trivial_torch_tools import Sequential, init, convert_each_arg, product
    from trivial_torch_tools.generics import to_pure, flatten
    
    # a cheating agent thats really good at creating trajectories
    class Agent(Skeleton):
        @enhance_with(EpisodeEnhancement, LoggerEnhancement,)
        def __init__(self,
            observation_space,
            response_space,
            responses=None,
            training=True,
            learning_rate=0.5,
            discount_factor=0.9,
            epsilon=1.0,
            epsilon_decay=0.0001,
            default_value_assumption=0,
            get_greedy_response=None,
            random_seed=None,
        ):
            self.has_water = None
            self.is_really_smart = randomly_pick_from([ True, False ])
            
            self.observation_space = observation_space
            self.observation_shape = self.observation_space.shape or (self.observation_space.n,)
            self.observation_size  = product(self.observation_shape)
            self.response_space    = response_space
            self.response_shape    = self.response_space.shape or (self.response_space.n,)
            self.response_size     = product(self.response_shape)
            self.learning_rate     = learning_rate
            self.discount_factor   = discount_factor
            self.responses         = OneHotifier(
                possible_values=(  responses or tuple(range(self.response_size))  ),
            )
            self.random_seed       = random_seed or time.time()
            self.training          = training
            self.epsilon           = epsilon        # Amount of randomness in the response selection
            self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
            self.debug             = LazyDict()
            
            self.default_value_assumption = default_value_assumption
            self._get_greedy_response       = get_greedy_response
        
        def get_position(self, timestep):
            for row_index, each_row in enumerate(timestep.observation.position):
                for column_index, each_cell in enumerate(each_row):
                    if each_cell:
                        return row_index, column_index
        
        def get_distance_score(self, new_position, ideal_positions=None):
            ideal_positions = self.ideal_positions if type(ideal_positions) == type(None)       else ideal_positions
                
            x, y = new_position
            minimum_distance = min(
                abs(x-good_x) + abs(y-good_y)
                    for good_x, good_y in ideal_positions
            )
            return minimum_distance
        
        def predicted_position(self, action):
            x, y = self.get_position()
            if action == env.actions.LEFT:
                return max(x-1,0), y
            elif action == env.actions.RIGHT:
                return min(x+1, world.grid_width-1), y
            elif action == env.actions.UP:
                return x, min(y+1, world.grid_height-1)
            elif action == env.actions.DOWN:
                return x, max(y-1, 0)
            else:
                raise Exception(f'''unknown action: {action}''')
        # 
        # mission hooks
        # 
        
        def when_mission_starts(self):
            pass
            
        def when_episode_starts(self):
            self.has_water = False
            self.has_visited_fire = False
            self.has_almost_visted_water = False
            self.fire_coordinates = all_argmax_coordinates(world.grid.fire)
            self.water_coordinates = all_argmax_coordinates(world.grid.water)
            self.ideal_positions = self.water_coordinates
            
        def when_timestep_starts(self):
            current_position = self.get_position()
            # update flags
            if not self.has_water:
                self.has_water = 0 == self.get_distance_score(new_position=current_position, ideal_positions=self.water_coordinates)
            if not self.has_almost_visted_water:
                self.has_almost_visted_water = 1 == self.get_distance_score(new_position=current_position, ideal_positions=self.water_coordinates)
            if not self.has_visited_fire:
                self.has_visited_fire = 0 == self.get_distance_score(new_position=current_position, ideal_positions=self.fire_coordinates)
                
            # keep ideal coordiates up to date
            if self.is_really_smart:
                self.ideal_positions = self.fire_coordinates if self.has_water   else  self.water_coordinate
            else:
                if (self.has_almost_visted_water and not self.has_visited_fire) or self.has_water:
                    self.ideal_positions = self.fire_coordinates
                else:
                    self.ideal_positions = self.water_coordinates
            # 
            # pick action that doesn't hurt the distance to the nearest ideal position
            # 
            current_distance_score = self.get_distance_score(current_position)
            new_score = -math.inf
            possible_responses = [ each for each in self.responses ]
            random.shuffle(possible_responses)
            for each_response in possible_responses:
                would_be_position = self.predicted_position(each_response)
                if self.get_distance_score(would_be_position) >= current_distance_score:
                    self.timestep.response = each_response
                    return
            # if all of them were bad, just pick one
            self.timestep.response = possible_responses[0]
            
        def when_timestep_ends(self):
            pass
            
        def when_episode_ends(self):
            pass
            
        def when_mission_ends(self):
            pass

    
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
print("# no_memory")
print("#")
with print.indent:
    best_score = 1
    while best_score == 1:
        # regenerate the timesteps until they require memory
        timesteps = []
        best_score = run_many_evaluations(iterations=1, competition_size=1, genetic_method="random", disable_memory=True)
    

# print("#")
# print("# genetic_mutations_with_perfect")
# print("#")
# with print.indent:
#     run_many_evaluations(iterations=5, genetic_method="mutation", disable_memory=False, enable_perfect=True)

# print("#")
# print("# genetic_mutations")
# print("#")
# with print.indent:
#     run_many_evaluations(iterations=25, genetic_method="mutation", disable_memory=False)

print("#")
print("# pure with perfect")
print("#")
with print.indent:
    run_many_evaluations(iterations=1, genetic_method="random", disable_memory=False, enable_perfect=True)

print("#")
print("# pure random")
print("#")
with print.indent:
    run_many_evaluations(iterations=5, genetic_method="random", disable_memory=False)