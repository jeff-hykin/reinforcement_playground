from collections import defaultdict
from copy import copy
import json
import pickle 
import random
import time

from blissful_basics import flatten_once, product, max_index, to_pure, flatten, print, FS, indent
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

number_of_timesteps    = 200
world_shape            = (9, 9)
action_length          = 2
memory_size            = 1
possible_memory_values = [0,1] # each bit is 1 or 0 
observation_size       = product(world_shape) + action_length
input_vector_size      = observation_size + memory_size
verbose                = False

# 
# definitions
# 
if True:
    class PrimaryState:
        def __init__(self, observation):
            self.observation = observation
            
            self.as_tuple = tuple(flatten([self.observation]))
    
        def __hash__(self): return hash(self.as_tuple)
    
    class DiscrepancyStateFormat:
        def __init__(self, observation, action):
            self.observation           = observation
            self.action                = action
            
            self.as_tuple = tuple(flatten([ self.observation, self.action ]))
        
        def __hash__(self): return hash(self.as_tuple)
    
    class MemoryState:
        def __init__(self, previous_memory_value, observation, action):
            self.previous_memory_value = previous_memory_value
            self.observation           = observation
            self.action                = action
            
            self.as_tuple = tuple(flatten([self.previous_memory_value, self.observation, self.action]))
        
        def __hash__(self): return hash(self.as_tuple)
        
    class RewardPredictionState:
        def __init__(self, observation, action, next_memory_value):
            self.observation           = observation
            self.action                = action
            self.next_memory_value     = next_memory_value
            
            self.as_tuple = tuple(flatten([self.observation, self.action, self.next_memory_value]))
        
        def __hash__(self): return hash(self.as_tuple)
    
# 
# discrepancy function
# 
if True:
    class Discrepancy(LazyDict):
        def __init__(self, *, state, trajectories_per_outcome=None):
            self.state = state
            self.trajectories_per_outcome = trajectories_per_outcome or LazyDict()
            
        @property
        def outcomes(self):
            return tuple(self.trajectories_per_outcome.keys())
        
        
    @print.indent.function_block
    def find_reward_discrepancies(trajectory, memory_agent):
        if not trajectory:
            trajectory = list(generate_samples(number_of_timesteps=number_of_timesteps))
        
        
        discrepancies = {}
        previous_index_for = defaultdict(lambda : 0)
        reward_predictor_table = {}
        memory_value = None
        was_last_step = True
        for phase in ["find_discrepancies", "find_all_trajectories_to_discrepancy_states"]:
            for training_index, each_timestep in enumerate(trajectory):
                if was_last_step: memory_value = None
                index             = each_timestep.index
                observation       = simplify_observation(each_timestep.observation)
                reaction          = simplify_reaction(each_timestep.reaction)
                reward            = each_timestep.reward
                is_last_step      = each_timestep.is_last_step
                hidden_info       = each_timestep.hidden_info
                episode_index     = each_timestep.hidden_info["episode_index"]
                discrepancy_state = each_timestep.hidden_info["discrepancy_state"] = DiscrepancyStateFormat(observation=observation, action=reaction)
                
                memory_state      = MemoryState(previous_memory_value=memory_value, observation=observation, action=reaction)
                is_known_discrepancy_state = discrepancy_state in discrepancies
                
                with print.indent.block(f"episode: {episode_index}"):
                    memory_value = memory_agent.get_next_memory_value(memory_state)
                    reward_state = RewardPredictionState(observation=observation, action=reaction, next_memory_value=memory_value)
                    if reward_state.as_tuple not in reward_predictor_table:
                        reward_predictor_table[reward_state.as_tuple] = reward
                        with print.indent.block(f"{training_index}: reward init"):
                            print(f'''reward_state = {reward_prediction_input_as_human_string(reward_state.as_tuple)}''')
                            print(f'''reward = {reward}''')
                    else:
                        predicted_reward = reward_predictor_table[reward_state.as_tuple]
                        was_wrong = predicted_reward != reward
                        if was_wrong and not is_known_discrepancy_state:
                            sub_trajectory = tuple(trajectory[0:index])
                            discrepancies[discrepancy_state] = Discrepancy(
                                state=DiscrepancyStateFormat(
                                    observation=observation,
                                    action=reaction
                                ),
                                trajectories_per_outcome=LazyDict({
                                    # reward is the outcome, and we've found a single trajectory for that outcome so far
                                    reward: set([ sub_trajectory ]),
                                })
                            )
                        with print.indent.block(f"{training_index}: reward check"):
                            print(f'''reward_state = {indent(reward_prediction_input_as_human_string(reward_state.as_tuple), by=4)}''')
                            print(f'''reward = {reward}''')
                            print(f'''predicted_reward: {predicted_reward}''')
                            print(f'''was_wrong: {was_wrong}''')
                    
                was_last_step = is_last_step
                
                # add every trajectory that leads to a discrepancy state
                if is_known_discrepancy_state:
                    outcome = reward
                    previous_index = previous_index_for[discrepancy_state]
                    discrepancy = discrepancies[discrepancy_state]
                    if outcome not in discrepancy:
                        discrepancy.trajectories_per_outcome[outcome] = set()
                    
                    sub_trajectory = tuple(trajectory[previous_index:index])
                    discrepancy.trajectories_per_outcome[outcome].add(sub_trajectory)
                    previous_index_for[discrepancy_state] = index
        
        return discrepancies


# 
# memory agent
# 
if True:
    _number_of_memory_agents = 0
    class MemoryAgent:
        def get_next_memory_value(self, memory_state):
            return [ False for each in range(memory_size) ]
    
    class MemoryTriggerAgent:
        observed_inputs = set()
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
            self.number_of_triggers = number_of_triggers
            self.triggers = []
        
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
                input_index : randomly_pick_from(set([ each_input[input_index] for each_input in MemoryTriggerAgent.observed_inputs ])) # TODO: this is pretty inefficient. Should pre-compute this when new inputs are observed and then just look it up
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
                memory_index : randomly_pick_from(possible_memory_values)
                    for memory_index in selected_memory_indicies 
            }
            
            return input_trigger_conditions, new_memory_mapping
    
        def get_next_memory_value(self, memory_state):
            input_vector = memory_state.as_tuple
            MemoryTriggerAgent.observed_inputs.add(input_vector)
            
            # little bit of a startup issue where triggers need to see a state before creating a trigger for it
            if self.is_stupid or len(MemoryTriggerAgent.observed_inputs) == 1:
                return [False] * memory_size
            else:
                # only make one trigger at a time
                if len(self.triggers) < self.number_of_triggers:
                    self.triggers.append(self.generate_random_trigger())
            
            for input_trigger_conditions, new_memory_mapping in self.triggers:
                failed_conditions = False
                for each_key, each_value in input_trigger_conditions.items():
                    if input_vector[each_key] != each_value:
                        failed_conditions = True
                        break
                if failed_conditions: continue
                        
                # if all the checks pass
                memory_copy = list(flatten(memory_state.previous_memory_value))
                for each_key, each_value in new_memory_mapping.items():
                    memory_copy[each_key] = each_value
                
                return tuple(memory_copy) 
            
            # one technically hardcoded trigger
            if type(memory_state.previous_memory_value) == type(None):
                return [False] * memory_size
            
            # if all triggers fail, preserve memory
            return tuple(memory_state.previous_memory_value)
            
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

    class MemoryHypothesisAgent:
        attempted_hypotheses_for = LazyDict() # keys are tuple(discrepancy_state, reward)
        observed_inputs = set()
        def __init__(self, id=None, is_stupid=False, **kwargs):
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
            # triggers
            # 
            self.trigger_map = LazyDict() # key=memory_state, value=set_of_states
            self.default_memory_value = next(iter(permutation_generator(memory_size, possible_values=possible_memory_values)))
        
        def switch_to_next_hypothesis(self, discrepancies):
            """
            returns input_trigger_conditions={ input_index : required_value }, new_memory_mapping={ memory_index: new output value }
            """
            # if there were no discrepancies, then no trigger is needed! perfect-as-far-as-we-can-tell agent. (will happen when primary agent hasn't explored enough)
            if len(discrepancies) == 0:
                return
            
            # since we're only solving the case of a single trigger at the moment, we're only going to consider the first discrepancy
            if True: # <- this will probably be a for loop later
                discrepancy = next(iter(discrepancies.values()))
                
                # 
                # check discrepancy size
                # 
                number_of_outcomes = len(discrepancy.trajectories_per_outcome)
                if number_of_outcomes > memory_size**len(possible_memory_values):
                    raise Exception(f'''Issue: for a particular discrepancy_state:\n    {discrepancy.state}\nThere were many possible outcomes: {list(discrepancy.trajectories_per_outcome.keys())}\nMore outcomes than can possibly fit in a memory of size {memory_size}^{possible_memory_values}''')
                memory_value_iterator = iter(permutation_generator(memory_size, possible_values=possible_memory_values))
                
                # 
                # find unions and intersections for each outcome
                # 
                union_for        = {}
                intersection_for = {}
                # remove one because it will be the default case (no exclusive trigger needed for it)
                default_outcome, *other_trajectories_per_outcome = tuple(discrepancy.trajectories_per_outcome.items())
                self.default_memory_value = tuple(next(memory_value_iterator))
                for each_outcome, sub_trajectories in other_trajectories_per_outcome:
                    intersection = None
                    union = set()
                    for each_trajectory in sub_trajectories:
                        states = set(
                            each_timestep.hidden_info["discrepancy_state"]
                                for training_index, each_timestep in each_trajectory
                        )
                        union        =     union    | states
                        intersection = intersection & states if not (intersection is None) else states
                    union_for[each_outcome]        = union
                    intersection_for[each_outcome] = intersection
                        
                # 
                # filter to find possible trigger states
                # 
                possible_descrepancy_state_triggers_for_outcome = {}
                for each_outcome, sub_trajectories in other_trajectories_per_outcome:
                    union_of_others = set(flatten(other_union for other_reward, other_union in union_for if other_reward != each_outcome))
                    possible_descrepancy_state_triggers_for_outcome[each_outcome] = intersection_for[each_outcome] - union_of_others
                
                # 
                # create trigger states for each non-default outcome
                #
                self.trigger_map.clear() # reset the old triggers which apparently didn't work cause there's still discrepancies
                for each_outcome, possible_states in possible_descrepancy_state_triggers_for_outcome.items():
                    situation_key = (discrepancy_state, each_outcome)
                    memory_value_for_this_situation = tuple(next(memory_value_iterator))
                    
                    # init if needed
                    if situation_key not in MemoryHypothesisAgent.attempted_hypotheses_for:
                        MemoryHypothesisAgent.attempted_hypotheses_for[situation_key] = set()
                    
                    # remove all the states we've already tried
                    remaining_possible_states = possible_states - MemoryHypothesisAgent.attempted_hypotheses_for[situation_key]
                    if len(remaining_possible_states) == 0:
                        raise Exception(f'''There's an issue with the {each_outcome}, as apparently there is no state that uniquely predicts it. Additional info:\n{first_discrepancy}''')
                    
                    # no particular order (heuristics could really help here)
                    triggering_descrepancy_state = next(iter(remaining_possible_states))
                    self.trigger_map[memory_value_for_this_situation] = set([ triggering_descrepancy_state ]) # note: simplification that only one state is the trigger
                    
                    # mark as attempted, so next time we can skip it
                    MemoryHypothesisAgent.attempted_hypotheses_for[situation_key].add(triggering_descrepancy_state)
                
            pass
    
        def get_next_memory_value(self, memory_state):
            # set the default case
            if type(memory_state.previous_memory_value) == type(None):
                return self.default_memory_value
                
            # redo this for self.trigger_map
            # FIXME: triggers need to be based on previous memory values as well
            discrepancy_state = DiscrepancyStateFormat(observation=memory_state.observation, action=memory_state.action)
            for each_memory_value, set_of_discrepancy_states in self.trigger_map.items():
                if discrepancy_state in set_of_discrepancy_states:
                    return each_memory_value
            
            # no triggers = preserve memory
            return memory_state.previous_memory_value
            
        def generate_mutated_copy(self, number_of_mutations):
            # just fully randomize it since its hard to mutate
            # TODO: change this in the future
            return MemoryHypothesisAgent()
        
        def __json__(self):
            # TODO
            pass
            # return {
            #     "class": "MemoryHypothesisAgent",
            #     "kwargs": dict(
            #         id=self.id,
            #         is_stupid=self.is_stupid,
            #         triggers=self.triggers,
            #     ),
            # }
        
        @staticmethod
        def from_json(json_data):
            # TODO
            pass
            # return MemoryHypothesisAgent(**json_data["kwargs"])

    
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
            
        
        def get_next_memory_value(self, memory_state):
            if self.is_stupid:
                return tuple([ False for each in range(memory_size) ])
            else:
                return self.table[memory_state.as_tuple]
            
        
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
        
        def get_next_memory_value(self, memory_state):
            agent_position_0 = next(iter(flatten(memory_state.observation))) # the first element in the observation
            if agent_position_0:
                memory_out = [ True ]
            else:
                memory_out = list(flatten(memory_state.previous_memory_value))
            
            key = memory_state.as_tuple
            self.table[key] = tuple([ not not each for each in memory_out ])
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
            timesteps = list(generate_samples(number_of_timesteps=number_of_timesteps))
        
        is_perfect_agent = isinstance(memory_agent, PerfectMemoryMapAgent)
        print(f"memory_agent: {memory_agent.id}")
        print(f'''is_perfect_agent = {is_perfect_agent}''')
        
        discrepancies = find_reward_discrepancies(trajectory=timesteps, memory_agent=memory_agent)
        
        score = ( len(timesteps)-len(discrepancies) ) / len(timesteps)
        print(f'''score = {score}''')
        return score

# 
# hypothesis machine
# 
if True:
    @print.indent.function_block
    def train_hypothesis_agent(*, iteration_count):
        memory_agent = MemoryHypothesisAgent()
        all_discrepancies = LazyDict()
        for progress, *_ in ProgressBar(iteration_count, title="generation"):
            with print.indent:
                trajectory    = generate_samples(number_of_timesteps=number_of_timesteps)
                discrepancies = find_reward_discrepancies(trajectory, memory_agent)
                print(f"number of discrepancies: {len(discrepancies)}")
                
                # 
                # merge them with the old trajectories
                # 
                for each_discrepancy_state, each_discrepancy in discrepancies.items():
                    if each_discrepancy_state not in all_discrepancies:
                        all_discrepancies[each_discrepancy_state] = each_discrepancy
                    else:
                        for each_outcome, new_sub_trajectories in each_discrepancy.items():
                            if each_outcome not in all_discrepancies[each_discrepancy_state].trajectories_per_outcome:
                                all_discrepancies[each_discrepancy_state].trajectories_per_outcome[each_outcome] = new_sub_trajectories
                            else:
                                # add any new sub_trajectories
                                all_discrepancies[each_discrepancy_state].trajectories_per_outcome[each_outcome] |= new_sub_trajectories
                
                progress.text = f"number of discrepancies: {len(discrepancies)}, trigger_map: {memory_agent.trigger_map}"
                
                # if there's still a problem, try to improve
                if len(discrepancies) > 0:
                    memory_agent.switch_to_next_hypothesis(all_discrepancies)
                
                
        

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
    
    for progress, *_ in ProgressBar(iterations, title=" generation"):
        if progress.index >= iterations:
            break
        
        # evaluate new ones
        with print.indent:
            for evaluation_progress, each_memory_agent in ProgressBar(next_generation, title=" evaluating this generation"):
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
            progress.pretext = compute_terminal_chart()
            
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
        grid_width=world_shape[0],
        grid_height=world_shape[1],
        visualize=False,
        # debug=True,
        fire_locations=[(-1,-1)],
        water_locations=[(0,0)],
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
            reaction_space,
            reactions=None,
            training=True,
            learning_rate=0.5,
            discount_factor=0.9,
            epsilon=1.0,
            epsilon_decay=0.0001,
            default_value_assumption=0,
            get_greedy_reaction=None,
            random_seed=None,
        ):
            self.has_water = None
            self.is_really_smart = randomly_pick_from([ True, False ])
            
            self.observation_space = observation_space
            self.observation_shape = self.observation_space.shape or (self.observation_space.n,)
            self.observation_size  = product(self.observation_shape)
            self.reaction_space    = reaction_space
            self.reaction_shape    = self.reaction_space.shape or (self.reaction_space.n,)
            self.reaction_size     = product(self.reaction_shape)
            self.learning_rate     = learning_rate
            self.discount_factor   = discount_factor
            self.reactions         = OneHotifier(
                possible_values=(  reactions or tuple(range(self.reaction_size))  ),
            )
            self.random_seed       = random_seed or time.time()
            self.training          = training
            self.epsilon           = epsilon        # Amount of randomness in the reaction selection
            self.epsilon_decay     = epsilon_decay  # Fixed amount to decrease
            self.debug             = LazyDict()
            
            self.scenic_route_propensity = 0.4 # 0.4==40% of the time, if the first not-bad option is a scenic route, it'll take it
            
            self.default_value_assumption = default_value_assumption
            self._get_greedy_reaction       = get_greedy_reaction
        
        def get_position(self, timestep=None):
            timestep = timestep or self.timestep
            for row_index, each_row in enumerate(timestep.observation.position):
                for column_index, each_cell in enumerate(each_row):
                    if each_cell:
                        return row_index, column_index
        
        def get_distance_between(self, new_position, ideal_positions=None):
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
                x -= 1
            elif action == env.actions.RIGHT:
                x += 1
            elif action == env.actions.DOWN:
                y -= 1
            elif action == env.actions.UP:
                y += 1
            else:
                raise Exception(f'''unknown action: {action}''')
            
            # stay in bounds
            if x > world.max_x_index: x = world.max_x_index
            if x < world.min_x_index: x = world.min_x_index
            if y > world.max_y_index: y = world.max_y_index
            if y < world.min_y_index: y = world.min_y_index
            
            return x, y
        # 
        # mission hooks
        # 
        
        def when_mission_starts(self):
            pass
            
        def when_episode_starts(self):
            self.has_water = False
            self.has_visited_fire = False
            self.has_almost_visted_water = False
            self.fire_coordinates = all_argmax_coordinates(world.state.grid.fire)
            self.water_coordinates = all_argmax_coordinates(world.state.grid.water)
            self.ideal_positions = self.water_coordinates
            
        def when_timestep_starts(self):
            current_position = self.get_position()
            # update flags
            if not self.has_water:
                self.has_water = 0 == self.get_distance_between(new_position=current_position, ideal_positions=self.water_coordinates)
            if not self.has_almost_visted_water:
                self.has_almost_visted_water = 1 == self.get_distance_between(new_position=current_position, ideal_positions=self.water_coordinates)
            if not self.has_visited_fire:
                self.has_visited_fire = 0 == self.get_distance_between(new_position=current_position, ideal_positions=self.fire_coordinates)
                
            # keep ideal coordiates up to date
            if self.is_really_smart:
                self.ideal_positions = self.fire_coordinates if self.has_water   else  self.water_coordinates
            else:
                if (self.has_almost_visted_water and not self.has_visited_fire) or self.has_water:
                    self.ideal_positions = self.fire_coordinates
                else:
                    self.ideal_positions = self.water_coordinates
            # 
            # pick action that doesn't hurt the distance to the nearest ideal position
            # 
            current_distance = self.get_distance_between(current_position)
            new_score = -math.inf
            possible_reactions = [ each for each in self.reactions ]
            random.shuffle(possible_reactions)
            fallback_option = possible_reactions[0]
            for each_reaction in possible_reactions:
                would_be_position = self.predicted_position(each_reaction)
                would_be_new_distance = self.get_distance_between(would_be_position)
                # skip all the definitely-bad options
                if would_be_new_distance > current_distance:
                    continue
                # immediately take good options
                elif would_be_new_distance < current_distance:
                    self.timestep.reaction = each_reaction
                    return
                else:
                    fallback_option = each_reaction
                    # only occasionally confirm neutral options
                    if random.random() < self.scenic_route_propensity:
                        self.timestep.reaction = each_reaction
                        return
            
            # if all of them were bad, just pick one
            self.timestep.reaction = fallback_option
            
        def when_timestep_ends(self):
            pass
            
        def when_episode_ends(self):
            pass
            
        def when_mission_ends(self):
            pass

    
    mr_bond = Agent(
        observation_space=env.observation_space,
        reaction_space=env.action_space,
        reactions=env.actions.values(),
    )
    
    # 
    # training
    # 
    for each_epsilon in [ 1.0 ]:
        mr_bond.epsilon = each_epsilon
        for progress, (episode_index, timestep) in ProgressBar(basic(agent=mr_bond, env=env), iterations=number_of_timesteps, title=" creating timesteps"):
            timestep.hidden_info = dict(episode_index=episode_index)
            offline_timesteps.append(timestep)
            world.random_seed = 1 # same world every time
            progress.text = f"""average_reward:{align(mr_bond.per_episode.average.reward, pad=4, decimals=0)}, reward: {align(mr_bond.episode.reward, digits=5, decimals=0)}, episode:{align(episode_index,pad=5)}, {align(mr_bond.epsilon*100, pad=3, decimals=0)}% random,\n{mr_bond.debug}"""
        
    return offline_timesteps


# 
# helpers
# 
if True:
    def reward_prediction_input_as_human_string(reward_prediction_input):
        *position, going_left, going_right, memory =  reward_prediction_input
        output = "\n"
        if going_left and going_right:
            output += "  ^  \n"
            output += "  |  \n"
        elif not going_left and not going_right:
            output += "  |  \n"
            output += "  v  \n"
        elif going_left:
            output += "     \n"
            output += " <-- \n"
        elif going_right:
            output += "     \n"
            output += " --> \n"
        
        output += "position = " + "".join([ ' __' if not each else f'{index}'.rjust(3) for index, each in enumerate(position)])+"\n"
        output += f"memory = [ {int(memory)} ]\n"
        return output

    def simplify_observation(observation):
        observation = tuple(flatten(to_pure(observation)))
        return observation[0:product(world_shape)]
    
    def simplify_reaction(action):
        if action == "UP":
            return ( True , True  )
        elif action == "DOWN":
            return ( False, False )
        elif action == "LEFT":
            return ( True , False )
        elif action == "RIGHT":
            return ( False, True )
        else:
            raise Exception(f'''
                action = {action}
                expected up/down/left/right
            ''')
        
print.flush.always = not verbose # False=>optimizes throughput, True=>optimizes responsiveness



train_hypothesis_agent(iteration_count=1000)

# print("#")
# print("# no_memory")
# print("#")
# with print.indent:
#     best_score = 1
#     while best_score == 1:
#         # regenerate the timesteps until they require memory
#         timesteps = []
#         number_of_timesteps *= 2 # scale up until it works
#         best_score = run_many_evaluations(iterations=1, competition_size=1, genetic_method="random", disable_memory=True)
    

# # print("#")
# # print("# genetic_mutations_with_perfect")
# # print("#")
# # with print.indent:
# #     run_many_evaluations(iterations=5, genetic_method="mutation", disable_memory=False, enable_perfect=True)

# # print("#")
# # print("# genetic_mutations")
# # print("#")
# # with print.indent:
# #     run_many_evaluations(iterations=25, genetic_method="mutation", disable_memory=False)

# print("#")
# print("# pure with perfect")
# print("#")
# with print.indent:
#     run_many_evaluations(iterations=1, competition_size=2, genetic_method="random", disable_memory=False, enable_perfect=True)

# print("#")
# print("# pure random")
# print("#")
# with print.indent:
#     run_many_evaluations(iterations=100, genetic_method="random", disable_memory=False)