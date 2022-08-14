from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from io import StringIO
from os import path
from random import random, randint, shuffle, seed
from typing import List, Optional
from warnings import warn
import math
from time import sleep, time

import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym import Env, spaces, utils
import torch
from torch import tensor
from blissful_basics import Object, product, is_required_by
from super_map import Map, LazyDict
from super_hash import super_hash
from trivial_torch_tools import to_tensor
from trivial_torch_tools.generics import to_pure

try: import atari_py
except ImportError as e: raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies by running 'pip install gym[atari]'.)".format(e))

import tools.universe.runtimes as runtimes
from world_builders.atari.environment import Environment

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
ACTION_TO_NUMBER = LazyDict({ value: key for key, value in ACTION_MEANING.items() })

class Discrete(spaces.Discrete):
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value

class World:
    def __init__(world, *, game="pong", mode=None, difficulty=None, obs_type="image", frameskip=(2, 5), repeat_action_probability=0.0, full_action_space=False, visualize=False, debug=False, random_seed=None):
        world._random_seed = time() if random_seed == None else random_seed
        world.visualize = visualize
        world.debug = debug
        world.reset()
        
        class Player1(gym.Env, utils.EzPickle):
            """
                The Atari Environment
                    (inherits from the openai gym environment: https://gym.openai.com/docs/)
                
                Environment(
                    game="pong",                   # use Environment.available_games to see available games
                    mode=None,                     # use Environment.available_modes_for(game) to see this list
                    difficulty=None,
                    obs_type="image",              # or "ram"
                    frameskip=(2, 5),              # random number between 2 and 5 
                    repeat_action_probability=0.0, # 0 means deterministic
                    full_action_space=False,
                )
            """
            actions = LazyDict({ key: key for key in ACTION_TO_NUMBER })
            metadata = {"render.modes": ["human", "rgb_array"]}
            available_games = atari_py.list_games()
            
            @classmethod
            def available_modes_for(game):
                ale = atari_py.ALEInterface()
                # load up the game
                ale.setInt(b"random_seed", world._random_seed)
                ale.loadROM(atari_py.get_game_path(game))
                return ale.getAvailableModes()

            @classmethod
            def _to_ram(ale):
                ram_size = ale.getRAMSize()
                ram = np.zeros((ram_size), dtype=np.uint8)
                ale.getRAM(ram)
                return ram
                
            def __init__(
                self,
                game="pong",
                mode=None,
                difficulty=None,
                obs_type="image",
                frameskip=(2, 5),
                repeat_action_probability=0.0,
                full_action_space=False,
            ):
                """
                Arguments:
                    game: the name of the game ("pong", "enduro", etc) dont add the "-v0"
                    mode: different modes are available for different games.
                    frameskip should be either a tuple (indicating a random range to choose from, with the top value exclude), or an int.
                """

                utils.EzPickle.__init__(
                    self, game, mode, difficulty, obs_type, frameskip, repeat_action_probability
                )
                assert obs_type in ("ram", "image")

                self.game = game
                self.game_path = atari_py.get_game_path(game)
                self.game_mode = mode
                self.game_difficulty = difficulty

                if not os.path.exists(self.game_path):
                    msg = "You asked for game %s but path %s does not exist"
                    raise IOError(msg % (game, self.game_path))
                self._obs_type = obs_type
                self.frameskip = frameskip
                self.ale = atari_py.ALEInterface()
                self.viewer = None

                # Tune (or disable) ALE's action repeat:
                # https://github.com/openai/gym/issues/349
                assert isinstance(
                    repeat_action_probability, (float, int)
                ), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
                self.ale.setFloat(
                    "repeat_action_probability".encode("utf-8"), repeat_action_probability
                )

                self.seed()

                self._action_set = (
                    self.ale.getLegalActionSet()
                    if full_action_space
                    else self.ale.getMinimalActionSet()
                )
                self.action_space = Discrete(len(self._action_set))

                (screen_width, screen_height) = self.ale.getScreenDims()
                if self._obs_type == "ram":
                    self.observation_space = spaces.Box(
                        low=0, high=255, dtype=np.uint8, shape=(128,)
                    )
                elif self._obs_type == "image":
                    self.observation_space = spaces.Box(
                        low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8
                    )
                else:
                    raise error.Error(
                        "Unrecognized observation type: {}".format(self._obs_type)
                    )

            def seed(self, seed=world._random_seed):
                self.np_random, seed1 = seeding.np_random(seed)
                # Derive a random seed. This gets passed as a uint, but gets
                # checked as an int elsewhere, so we need to keep it below
                # 2**31.
                seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
                # Empirically, we need to seed before loading the ROM.
                self.ale.setInt(b"random_seed", seed2)
                self.ale.loadROM(self.game_path)

                if self.game_mode is not None:
                    modes = self.ale.getAvailableModes()

                    assert self.game_mode in modes, (
                        'Invalid game mode "{}" for game {}.\nAvailable modes are: {}'
                    ).format(self.game_mode, self.game, modes)
                    self.ale.setMode(self.game_mode)

                if self.game_difficulty is not None:
                    difficulties = self.ale.getAvailableDifficulties()

                    assert self.game_difficulty in difficulties, (
                        'Invalid game difficulty "{}" for game {}.\nAvailable difficulties are: {}'
                    ).format(self.game_difficulty, self.game, difficulties)
                    self.ale.setDifficulty(self.game_difficulty)

                return [seed1, seed2]
            
            @is_required_by(gym)
            def step(self, a):
                reward = 0.0
                action = self._action_set[a]

                if isinstance(self.frameskip, int):
                    num_steps = self.frameskip
                else:
                    num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
                for _ in range(num_steps):
                    reward += self.ale.act(action)
                ob = self._get_obs()

                return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

            def _get_image(self):
                return self.ale.getScreenRGB2()

            def _get_ram(self):
                return Environment._to_ram(self.ale)

            @property
            def _n_actions(self):
                return len(self._action_set)

            def _get_obs(self):
                if self._obs_type == "ram":
                    return self._get_ram()
                elif self._obs_type == "image":
                    img = self._get_image()
                return img

            # return: (states, observations)
            @is_required_by(gym)
            def reset(self):
                self.ale.reset_game()
                return self._get_obs()

            def render(self, mode="human"):
                img = self._get_image()
                if mode == "rgb_array":
                    return img
                elif mode == "human":
                    from gym.envs.classic_control import rendering

                    if self.viewer is None:
                        self.viewer = rendering.SimpleImageViewer()
                    self.viewer.imshow(img)
                    return self.viewer.isopen

            def close(self):
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None

            def get_action_meanings(self):
                return [ACTION_MEANING[i] for i in self._action_set]

            def get_keys_to_action(self):
                KEYWORD_TO_KEY = {
                    "UP": ord("w"),
                    "DOWN": ord("s"),
                    "LEFT": ord("a"),
                    "RIGHT": ord("d"),
                    "FIRE": ord(" "),
                }

                keys_to_action = {}

                for action_id, action_meaning in enumerate(self.get_action_meanings()):
                    keys = []
                    for keyword, key in KEYWORD_TO_KEY.items():
                        if keyword in action_meaning:
                            keys.append(key)
                    keys = tuple(sorted(keys))

                    assert keys not in keys_to_action
                    keys_to_action[keys] = action_id

                return keys_to_action

            def clone_state(self):
                """Clone emulator state w/o system state. Restoring this state will
                *not* give an identical environment. For complete cloning and restoring
                of the full state, see `{clone,restore}_full_state()`."""
                state_ref = self.ale.cloneState()
                state = self.ale.encodeState(state_ref)
                self.ale.deleteState(state_ref)
                return state

            def restore_state(self, state):
                """Restore emulator state w/o system state."""
                state_ref = self.ale.decodeState(state)
                self.ale.restoreState(state_ref)
                self.ale.deleteState(state_ref)

            def clone_full_state(self):
                """Clone emulator state w/ system state including pseudorandomness.
                Restoring this state will give an identical environment."""
                state_ref = self.ale.cloneSystemState()
                state = self.ale.encodeState(state_ref)
                self.ale.deleteState(state_ref)
                return state

            def restore_full_state(self, state):
                """Restore emulator state w/ system state including pseudorandomness."""
                state_ref = self.ale.decodeState(state)
                self.ale.restoreSystemState(state_ref)
                self.ale.deleteState(state_ref)
        
        class Player(Env):
            actions = LazyDict(dict(
                LEFT  = "LEFT",
                UP    = "UP",
                DOWN  = "DOWN",
                RIGHT = "RIGHT",
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
                
                # for openai gym
                self.observation_shape = (1,1)
                self.observation_space = spaces.Box(low=torch.zeros_like(world.state.grid).numpy(), high=torch.ones_like(world.state.grid).numpy(), dtype=np.float16)
                self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([1,1]) )
            
            @property
            def position(self):
                return world.state.position_of[self]
            
            @property
            def observation(self):
                return world.state.grid
                
            def compute_reward(self):
                fires_before = self.previous_observation.fire.sum()
                fires_now = self.observation.fire.sum()
                
                if fires_before > fires_now:
                    return 50
                else:
                    # penalize hitting a wall
                    if to_pure(self.previous_observation) == to_pure(self.observation):
                        return -20
                    else:
                        return -1
            
            def check_for_done(self):
                fires_now = self.observation.fire.sum()
                
                if fires_now == 0:
                    return True
                else:
                    return False
            
            def perform_action(self, action):
                action = to_pure(action)
                if isinstance(action, (list, tuple)):
                    # convert to boolean
                    action = tuple(not not each for each in action)
                    if action == (True, False):
                        action = "LEFT"
                    elif action == (False, True):
                        action = "RIGHT"
                    elif action == (True, True):
                        action = "UP"
                    elif action == (False, False):
                        action = "DOWN"
                self.previous_observation = deepcopy(self.observation)
                self.previous_action      = deepcopy(self.action)
                self.action = action
                world.request_change(self, action)
            
            def step(self, action):
                self.perform_action(action)
                next_state = self.observation
                reward     = self.compute_reward()
                done       = self.check_for_done()
                debug_info = LazyDict(has_water=world.state.has_water[self], position=self.position)
                
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
        # TODO:
        return ""
    
    @property
    def random_seed(self):
        return self._random_seed
    
    @random_seed.setter
    def random_seed(self, value):
        self._random_seed = value
    
    def reset(world):
        pass
    
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
        if new_position.x > world.max_x_index: new_position.x = world.max_x_index
        if new_position.x < world.min_x_index: new_position.x = world.min_x_index
        if new_position.y > world.max_y_index: new_position.y = world.max_y_index
        if new_position.y < world.min_y_index: new_position.y = world.min_y_index
        
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
        if world.debug: print(f'''new_position = {new_position}''')
        if world.debug: print(f'''has_water = {has_water}''')
        if world.debug: print(f'''fire_status = {fire_status}''')
        if world.debug: sleep(0.5)
        
        if world.visualize:
            print(world)
            sleep(0.7)
        # request granted
        return True