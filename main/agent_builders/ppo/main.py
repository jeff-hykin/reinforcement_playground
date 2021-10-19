import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

# local 
from tools.all_tools import PATHS, product, FS, Countdown
from tools.reinverse import ConnectBody

from agent_builders.ppo.rollout_buffer import RolloutBuffer
from agent_builders.ppo.actor_critic import ActorCritic
from agent_builders.ppo.ppo_brain import PpoBrain

@ConnectBody
class AgentBuilder:
    def __init__(
        self,
        body,
        timesteps_before_weight_update=1600,
        episodes_before_checkpoint=400,
        timesteps_before_standard_deviation_decay=160,
        action_standard_deviation_decay_rate=0.1,
        minimum_action_standard_deviation=0.0001,
        save_folder="./logs/ppo/",
        **kwargs
    ):
        self.body = body
        self.action_space = self.body.action_space
        self.observation_space = self.body.observation_space
        self.countdown_till_update     = Countdown(size=timesteps_before_weight_update)
        self.countdown_till_checkpoint = Countdown(size=episodes_before_checkpoint)
        self.countdown_till_decay      = Countdown(size=timesteps_before_standard_deviation_decay)
        self.action_standard_deviation_decay_rate      = action_standard_deviation_decay_rate
        self.minimum_action_standard_deviation         = minimum_action_standard_deviation
        self.save_folder                               = save_folder
        
        self.brain = PpoBrain(
            state_dim=product(self.observation_space.shape or [self.observation_space.n]),
            action_dim=product(self.action_space.shape or [self.action_space.n]),
            has_continuous_action_space=(not isinstance(self.action_space, gym.spaces.Discrete)),
            **kwargs,
        )
    
    @ConnectBody.when_mission_starts
    def when_mission_starts(self):
        pass
        
    @ConnectBody.when_episode_starts
    def when_episode_starts(self, episode_index):
        self.episode_index = 0
        self.accumulated_reward = 0
        print(f'episode: {episode_index}')
    
    @ConnectBody.when_timestep_happens
    def when_timestep_happens(self, timestep_index):
        # 
        # save the reward for the previous action
        # 
        reward = self.body.get_reward()
        self.accumulated_reward += reward
        self.brain.buffer.rewards.append(reward)
        self.brain.buffer.is_terminals.append(False)
        
        # 
        # occasionally update the network
        # 
        if self.countdown_till_update():
            self.brain.update()
        
        # 
        # update action standard deviation
        # 
        # if continuous action space; then decay action std of ouput action distribution
        if self.brain.has_continuous_action_space:
            if self.countdown_till_decay():
                self.brain.decay_action_std(self.action_standard_deviation_decay_rate, self.minimum_action_standard_deviation)
        
        # 
        # take action!
        # 
        observation = self.body.get_observation()
        # brain decides action (reshape because the agent wants a batch)
        action_choice = self.brain.select_action(observation.reshape((1,*observation.shape)))
        # actually perform the action in the world
        self.body.perform_action(action_choice)
        
    @ConnectBody.when_episode_ends
    def when_episode_ends(self, episode_index):
        # 
        # save last reward
        # 
        reward = self.body.get_reward()
        self.accumulated_reward += reward
        self.brain.buffer.rewards.append(reward)
        self.brain.buffer.is_terminals.append(True)
        
        #
        # occasionally save the model
        #
        if self.countdown_till_checkpoint():
            print("saving model in " + self.save_folder)
            self.save_network()
            print("model saved")
    
    @ConnectBody.when_mission_ends
    def when_mission_ends(self):
        # 
        # save the network
        # 
        print("saving model in " + self.save_folder)
        self.save_network()
        print("model saved")
        
    
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
        if overwrite_previous:
            FS.delete(save_point)
        self.brain.save(save_point)


# for testing:
#     from world_builders.atari.main import WorldBuilder
#     atari = WorldBuilder(game="enduro")
#     ppo_brain = AgentBuilder(body=atari.bodies[0])