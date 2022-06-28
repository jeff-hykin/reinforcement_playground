from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import math
from random import random, randint, shuffle
from warnings import warn
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
from blissful_basics import Object, product
from super_map import LazyDict
from torch import tensor
from super_hash import super_hash

from gym import Env, spaces, utils

from trivial_torch_tools import to_tensor
from trivial_torch_tools.generics import to_pure


layers_enum = LazyDict(dict(
    position=0,
    water=1,
    fire=2,
))

class Position(list):
    @property
    def x(self): return self[0]
    
    @x.setter
    def x(self, value): self[0] = value
    
    @property
    def y(self): return self[1]
    
    @y.setter
    def y(self, value): self[1] = value
    
    @property
    def z(self): return self[2]
    
    @z.setter
    def z(self, value): self[2] = value
    
    def __repr__(self):
        return f'(x={self.x},y={self.y},z={self.z})'
    
def generate_random_map(size):
    if size % 2 == 0:
        raise Exception(f'''size must be odd when generating a map''')
    
    shape = (len(layers_enum), size, size)
    min_index = 0
    max_index = size-1
    layers = torch.zeros(shape) != 0 # boolean tensor with default of false
    for each_key, each_value in layers_enum.items():
        setattr(layers, each_key, layers[each_value])
    start_x = randint(min_index+1,max_index-1)
    start_y = randint(min_index+1,max_index-1)
    center_square_location = (start_x, start_y)
    
    # 
    # use dividing line
    # 
    possible_fire_positions = []
    possible_water_positions = []
    if random() > 0.5:
        # use start_x as dividing line
        for x_index, each_row in enumerate(layers.position):
            for y_index, each_cell in enumerate(each_row):
                if x_index > start_x:
                    possible_fire_positions.append((x_index, y_index))
                elif x_index < start_x:
                    possible_water_positions.append((x_index, y_index))
    else:
        # use start_y as dividing line
        for x_index, each_row in enumerate(layers.position):
            for y_index, each_cell in enumerate(each_row):
                if y_index > start_y:
                    possible_fire_positions.append((x_index, y_index))
                elif y_index < start_y:
                    possible_water_positions.append((x_index, y_index))
    
    # set current location
    layers.position[start_x, start_y] = True
    
    # generate fires
    number_of_fires_minus_one = randint(0, round(len(possible_fire_positions)/2))
    shuffle(possible_fire_positions)
    for each_fire_index, (x,y) in enumerate(possible_fire_positions):
        if each_fire_index > number_of_fires_minus_one:
            break
        else:
            layers.fire[x, y] = True
    
    # generate waters
    number_of_waters_minus_one = randint(0, round(len(possible_water_positions)/2))
    shuffle(possible_water_positions)
    for each_water_index, (x, y) in enumerate(possible_water_positions):
        if each_water_index > number_of_waters_minus_one:
            break
        else:
            layers.water[x, y] = True
        
    number_of_states = (
        product(layers.position.shape) # number of positions the player can be in
        * (2 ** product(layers.fire.shape)) # squares cannot be both a fire and a water square, so we treat them as binary. This should still be an overestimate of true possible number of states
    )
    return layers, Position((start_x, start_y)), number_of_states

class Discrete(spaces.Discrete):
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value

class World:
    def __init__(world, *, grid_size, visualize=False, debug=False):
        world.visualize = visualize
        world.debug = debug
        world.grid_size = grid_size
        world.reset()
        
        class Player(Env):
            reward_range = (0,100)
            actions = LazyDict(dict(
                LEFT  = "LEFT",
                DOWN  = "DOWN",
                RIGHT = "RIGHT",
                UP    = "UP",
            ))
            action_space      = Discrete(len(actions))
            observation_space = Discrete(world.number_of_grid_states)
            observation_space.shape = tuple(to_tensor(world.state.grid).shape)
            
            def __init__(self):
                world.state.has_water[self] = False
                world.state.position_of[self] = world.start_position
                self.previous_action = None
                self.previous_observation = None
                self.action = None
            
            @property
            def position(self):
                return world.state.position_of[self]
            
            @property
            def observation(self):
                return world.state.grid
                
            
            def check_for_reward(self):
                if super_hash(self.observation) == super_hash(self.previous_observation):
                    return 0
                
                fires_before = self.previous_observation.fire.sum()
                fires_now = self.observation.fire.sum()
                
                if fires_before > fires_now:
                    return 50
                else:
                    # penalize hitting a wall
                    if to_pure(self.previous_action) == to_pure(self.action) and to_pure(self.previous_observation) == to_pure(self.observation):
                        return 0
                    else:
                        return 5
            
            def check_for_done(self):
                fires_now = self.observation.fire.sum()
                
                if fires_now == 0:
                    return True
                else:
                    return False
            
            def perform_action(self, action):
                self.previous_observation = deepcopy(self.observation)
                for each_key, each_value in layers_enum.items():
                    setattr(self.previous_observation, each_key, self.previous_observation[each_value])
                self.previous_action = deepcopy(self.action)
                self.action = action
                world.request_change(self, action)
            
            def step(self, action):
                self.perform_action(action)
                next_state = self.observation
                reward     = self.check_for_reward()
                done       = self.check_for_done()
                debug_info = Object(has_water=world.state.has_water[self], position=self.position)
                
                if world.visualize:
                    print(f'''player1 reward = {f"{reward}".rjust(3)}, done = {done}''')
                
                return next_state, reward, done, debug_info

            def reset(self,):
                # ask the world to reset
                world.request_change(self, World.reset)
                self.__init__()
                return self.observation
            
            def close(self):
                pass
            
            def __repr__(self):
                return world.__repr__()

        
        world.Player = Player
    
    def __repr__(world):
        output = ""
        for x, each_row in enumerate(world.state.grid.position):
            output += f'-----'*len(each_row)+'-\n'
            # add all the fires
            for y, _ in enumerate(each_row):
                output += f'|  🔥' if world.state.grid.fire[x,y] else f'|    '
            output += f'|\n'
            # add player and faucet
            for y, _ in enumerate(each_row):
                person_space = '🏃‍' if world.state.grid.position[x,y] else '  '
                water_space  = '🚰' if world.state.grid.water[x,y] else '  '
                output += f'|{person_space}{water_space}'
            output += f'|\n'
        output += f'-----'*len(each_row)+'-\n'
        return output
    
    def reset(world):
        world.state = Object(
            grid=None,
            has_water={},
            position_of={},
        )
        world.state.grid, world.start_position, world.number_of_grid_states = generate_random_map(world.grid_size)
        world.min_index = 0
        world.max_index = world.grid_size-1
        world.number_of_states = world.number_of_grid_states + 1
        
        world.has_water = defaultdict(lambda : False)
    
    def request_change(world, player, change):
        if change == World.reset:
            world.reset()
            return True
        # 
        # compute changes
        # 
        
        old_position = Position(player.position)
        new_position = Position(old_position)
                
        if change == player.actions.LEFT:
            new_position.x -= 1
        elif change == player.actions.RIGHT:
            new_position.x += 1
        elif change == player.actions.DOWN:
            new_position.y -= 1
        elif change == player.actions.UP:
            new_position.y += 1
        else:
            warn(f"invalid change ({change}) was selected, ignoring")
        
        # stay in bounds
        if new_position.x > world.max_index: new_position.x = world.max_index
        if new_position.x < world.min_index: new_position.x = world.min_index
        if new_position.y > world.max_index: new_position.y = world.max_index
        if new_position.y < world.min_index: new_position.y = world.min_index
        
        # check if player has water
        has_water = world.state.has_water[player] or world.state.grid.water[new_position.x, new_position.y]
        
        # use water automatically
        fire_status = world.state.grid.fire[new_position.x, new_position.y] and not has_water
            
        
        # 
        # mutate state
        # 
        if world.debug: print(world)
        
        # position
        world.state.position_of[player] = new_position
        # grid
        world.state.grid.position[tuple(old_position)] = False
        world.state.grid.position[tuple(new_position)] = True
        # has water
        world.state.has_water[player] = has_water
        # put out fire
        world.state.grid.fire[new_position.x, new_position.y] = fire_status
        
        if world.debug: print(world)
        if world.debug: print(f'''has_water = {has_water}''')
        if world.debug: print(f'''fire_status = {fire_status}''')
        if world.debug: from time import sleep 
        if world.debug: sleep(0.5)
        
        if world.visualize:
            from time import sleep
            print(f'''world = {world}''')
            sleep(0.7)
        # request granted
        return True