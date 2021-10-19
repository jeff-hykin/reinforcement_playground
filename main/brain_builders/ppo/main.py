import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

# local 
from tools.all_tools import PATHS, product, FS
from tools.reinverse import ConnectBody

from brain_builders.ppo.rollout_buffer import RolloutBuffer
from brain_builders.ppo.actor_critic import ActorCritic
from brain_builders.ppo.ppo_agent import PpoAgent

@ConnectBody
class Brain:
    def __init__(
        self,
        body,
        timesteps_before_weight_update=1600,
        timesteps_before_standard_deviation_decay=160,
        action_standard_deviation_decay_rate=0.1,
        minimum_action_standard_deviation=0.0001,
        save_folder="./logs/ppo/",
        **kwargs
    ):
        self.body = body # call it self.whatever_you_want, just need a body attribute
        self.action_space = self.body.action_space
        self.observation_space = self.body.observation_space
        self.timesteps_before_weight_update            = timesteps_before_weight_update
        self.timesteps_before_standard_deviation_decay = timesteps_before_standard_deviation_decay
        self.action_standard_deviation_decay_rate      = action_standard_deviation_decay_rate
        self.minimum_action_standard_deviation         = minimum_action_standard_deviation
        self.save_folder                               = save_folder
        self.agent = PPO(
            state_dim=product(self.observation_space.shape),
            action_dim=product(self.action_space.shape),
            has_continuous_action_space=(not isinstance(self.action_space, gym.spaces.Discrete)),
            **kwargs,
        )
    
    @ConnectBody.when_mission_starts
    def _(self, episode_index):
        # a simple counter
        self.remaining_timesteps_before_update = self.timesteps_before_weight_update
        self.remaining_timesteps_before_standard_deviation_decay = self.timesteps_before_standard_deviation_decay
        
    @ConnectBody.when_episode_starts
    def _(self, episode_index):
        self.episode_index = 0
        self.accumulated_reward = 0
    
    @ConnectBody.when_timestep_happens
    def _(self, timestep_index):
        # 
        # save the reward for the previous action
        # 
        if timestep_index != 0:
            reward = self.body.get_reward()
            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(False)
        # 
        # occasionally update the network
        # 
        self.remaining_timesteps_before_update -= 1
        if self.remaining_timesteps_before_update <= 0:
            # reset the counter
            self.remaining_timesteps_before_update = self.timesteps_before_weight_update
            self.agent.update()
        
        # 
        # update action standard deviation
        # 
        # if continuous action space; then decay action std of ouput action distribution
        if self.agent.has_continuous_action_space:
            self.remaining_timesteps_before_standard_deviation_decay -= 1
            if self.remaining_timesteps_before_standard_deviation_decay == 0:
                # reset the counter
                self.remaining_timesteps_before_standard_deviation_decay = self.timesteps_before_standard_deviation_decay
                self.agent.decay_action_std(self.action_standard_deviation_decay_rate, self.minimum_action_standard_deviation)
        
        # 
        # take action!
        # 
        observation = self.body.get_observation()
        # brain decides action
        action_choice = self.agent.select_action(observation)
        # actually perform the action in the world
        self.body.take_action(action_choice)
        
    @ConnectBody.when_episode_ends
    def _(self, episode_index):
        # 
        # save last reward
        # 
        self.agent.buffer.rewards.append(self.body.get_reward())
        self.agent.buffer.is_terminals.append(True)
        # (no action/observation logic needed)
    
    @ConnectBody.when_mission_ends
    def _(self, episode_index):
        # 
        # save the network
        # 
        self.save_network()
        
    
    # helper
    def save_network(self, overwrite_previous=False):
        # make sure the folder exists
        FS.touch_dir(self.save_folder)
        # find the biggest number
        max_number = 0
        for each in FS.list_files(self.save_folder):
            *folders, file_name, file_extension = FS.path_pieces(each)
            next_number = 0
            try:
                next_number = int(file_name)
            except Exception as error:
                pass
            max_number = max(next_number, max_number)
        
        if not overwrite_previous:
            max_number += 1
        save_point = FS.join(self.save_folder, str(max_number)+".model")
        print("saving model at : " + save_point)
        ppo_agent.save(save_point+)
        print("model saved")
        