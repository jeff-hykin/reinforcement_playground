import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import collections 
import cv2
import time
import include
from super_map import LazyDict, Map
from statistics import mean as average

from tools.agent_skeleton import Skeleton
from tools.file_system_tools import FileSystem
from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, torch_image_to_opencv_image, to_tensor, init, forward, Sequential, tensor_to_image
from tools.basics import to_pure
from tools.agent_recorder import AgentRecorder

def observation_generator():
    database = AgentRecorder(
        save_to="resources/datasets.ignore/atari/baselines_pretrained@breakout_custom"
    )
    generator = database.load_batch_data("balanced64", epochs=9999999)
    for observations, actions in generator:
        for each in observations:
            yield each

observation_iterator = observation_generator()
                    
class Agent(Skeleton):
    @init.hardware
    def __init__(self, observation_space, action_space, **config):
        # 
        # special
        # 
        self.observation     = None # external world will change this
        self.reward          = None # external world will change this
        self.action          = None # external world will read this
        self.episode_is_over = None # external world will change this
        
        # 
        # regular/misc attributes
        # 
        self.which_model       = config.get("which_model", "")
        AutoImitator = include.file(f"./models/{self.which_model}", {"__file__":__file__}).AutoImitator
        
        # import [name] from-anywhere (doesnt pollute global namespace)
        hello = include.file("./path/to/file/with/hello/func/code.py", {"__file__":__file__}).hello
        self.path              = config.get("path", None)
        self.random_proportion = config.get("random_proportion", None)
        self.logging = Agent.Logger(agent=self, **config)
        self.model = AutoImitator().to(self.hardware)
        
        
        self.action_from_given_observation = Map({0:0,1:0,2:0,3:0})
        self.action_from_database_observation = Map({0:0,1:0,2:0,3:0})
        self.observation_iterator = observation_iterator

    class Logger:
        # depends on:
        #     self.agent.reward
        #     self.agent.loss
        def __init__(self, agent, **config):
            self.agent = agent
            
            self.should_print     = config.get("should_print"    , True)
            self.should_display   = config.get("should_display"  , False)
            self.live_updates     = config.get("live_updates"    , False)
            self.smoothing_amount = config.get("smoothing_amount", 5    )
            self.across_episodes = LazyDict(
                average_reward=None,
            )
            self.episode_rewards = []
            self.episode_losses  = []
            self.episode_reward_card = None
            self.episode_loss_card = None
            self.number_of_updates = 0
            self.action_frequency = Map({0:0,1:0,2:0,3:0})
            self.reward_frequency = Map()
            
            # init class attributes if doesn't already have them
            self.static = Agent.Logger.static = LazyDict(
                agent_number_count=0,
                total_number_of_episodes=0,
                total_number_of_timesteps=0,
                start_time=time.time(),
            ) if not hasattr(Agent.Logger, "static") else Agent.Logger.static
            
            # agent number count
            self.static.agent_number_count += 1
            self.agent_number = self.static.agent_number_count
            
        def when_mission_starts(self):
            self.episode_rewards.clear()
            self.episode_losses.clear()
            if self.live_updates:
                self.episode_loss_card = ss.DisplayCard("quickLine",[])
                ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Loss, ➡️ Per Episode")
                self.episode_reward_card = ss.DisplayCard("quickLine",[])
                ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Rewards, ➡️ Per Episode")
            
        def when_episode_starts(self, episode_index):
            self.accumulated_reward = 0
            self.accumulated_loss   = 0
            self.static.total_number_of_episodes += 1
        
        def when_timestep_starts(self, timestep_index):
            self.static.total_number_of_timesteps += 1
            
        def when_timestep_ends(self, timestep_index):
            self.action_frequency[self.agent.action] += 1
            self.reward_frequency[self.agent.reward] += 1
            self.accumulated_reward += self.agent.reward
        
        def when_episode_ends(self, episode_index):
            # logging
            self.episode_rewards.append(self.accumulated_reward)
            self.episode_losses.append(self.accumulated_loss)
            if self.live_updates:
                self.episode_reward_card.send([episode_index, self.accumulated_reward      ])
                self.episode_loss_card.send([episode_index, self.accumulated_loss  ])
            self.across_episodes.average_reward = average(self.episode_rewards)
            if self.should_print:
                print("\033[2J\033[1;1H") # clearscreen 
                print('episode_index = ', episode_index)
                print(f'    total_number_of_timesteps :{self.static.total_number_of_timesteps}',)
                print(f'    number_of_updates         :{self.number_of_updates}',)
                print(f'    average_episode_time      :{(time.time()-self.static.start_time)/self.static.total_number_of_episodes}',)
                print(f'    average_reward            :{self.across_episodes.average_reward     }',)
                print(f'    accumulated_reward        :{self.accumulated_reward      }',)
                print(f'    accumulated_loss          :{self.accumulated_loss  }',)
                print(f'    action_frequency          :{self.agent.action_from_given_observation  }',)
                print(f'    reward_frequency          :{self.reward_frequency }',)
        
        def when_mission_ends(self,):
            if self.should_display:
                # graph reward results
                ss.DisplayCard("quickLine", stat_tools.rolling_average(self.episode_losses, self.smoothing_amount))
                ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Losses Per Episode")
                ss.DisplayCard("quickLine", stat_tools.rolling_average(self.episode_rewards, self.smoothing_amount))
                ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Rewards Per Episode")
        
        def when_weight_update_starts(self):
            self.number_of_updates += 1

        def when_weight_update_ends(self):
            self.accumulated_loss += self.agent.loss.item()
        
        @property
        def action_percents(self):
            return stat_tools.proportionalize(self.action_frequency[Map.Dict])
        
    
    # 
    # Hooks (Special Names)
    # 
    def when_mission_starts(self):
        self.logging.when_mission_starts()
        
    def when_episode_starts(self, episode_index):
        self.logging.when_episode_starts(episode_index)
        
    def when_timestep_starts(self, timestep_index):
        self.logging.when_timestep_starts(timestep_index)
        # 
        # run the model
        #
        if (
                   False
                or (random.random() <= self.random_proportion)
                # or (timestep_index < 10000 and random.random() <= 0.009)
                # or (timestep_index < 1000 and random.random() <= 0.02)
                # or (timestep_index < 100  and random.random() <= 0.05)
                # or (timestep_index < 10   and random.random() <= 0.1)
                # or (timestep_index < 2)
            ):
            self.logging.action_frequency = Map()
            action_one_hot = self.model.forward(
                to_tensor(next(self.observation_iterator))
            )
        else:
            self.logging.action_frequency = self.action_from_given_observation
            action_one_hot = self.model.forward(
                to_tensor(self.observation)
            )
        
        # 
        # choose an action
        # 
        self.action = to_pure(action_one_hot.argmax())
        
    def when_timestep_ends(self, timestep_index):
        self.logging.when_timestep_ends(timestep_index)
    
    def when_episode_ends(self, episode_index):
        self.logging.when_episode_ends(episode_index)
    
    def when_mission_ends(self):
        self.logging.when_mission_ends()
    