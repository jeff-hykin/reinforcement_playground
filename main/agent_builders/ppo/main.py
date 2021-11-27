import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from super_map import LazyDict
from tools.record_keeper import RecordKeeper

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
        # Agent parameters
        body,
        episodes_before_checkpoint=400,
        timesteps_before_weight_update=1600,
        timesteps_before_standard_deviation_decay=160,
        action_standard_deviation_decay_rate=0.1,
        minimum_action_standard_deviation=0.0001,
        save_folder="./logs/ppo/",
        record_keeper=None,
        # Model parameters
        actor_learning_rate=0.0003,
        critic_learning_rate=0.001,
        discount_factor=0.99,
        number_of_epochs_to_optimize=40,
        loss_clamp_boundary=0.2,
        action_std_init=0.6,
    ):
        # Copy args to self
        self.body                                      = body
        self.episodes_before_checkpoint                = episodes_before_checkpoint
        self.timesteps_before_weight_update            = timesteps_before_weight_update
        self.timesteps_before_standard_deviation_decay = timesteps_before_standard_deviation_decay
        self.action_standard_deviation_decay_rate      = action_standard_deviation_decay_rate
        self.minimum_action_standard_deviation         = minimum_action_standard_deviation
        self.save_folder                               = save_folder
        self.record_keeper                             = record_keeper
        self.actor_learning_rate                       = actor_learning_rate
        self.critic_learning_rate                      = critic_learning_rate
        self.discount_factor                           = discount_factor
        self.number_of_epochs_to_optimize              = number_of_epochs_to_optimize
        self.loss_clamp_boundary                       = loss_clamp_boundary
        self.action_std_init                           = action_std_init
        
        # Save config
        self.action_space                              = self.body.action_space
        self.observation_space                         = self.body.observation_space
        self.is_continuous_action_space                = not isinstance(self.action_space, gym.spaces.Discrete)
        
        # setup countdowns
        self.countdown_till_update     = Countdown(size=timesteps_before_weight_update)
        self.countdown_till_checkpoint = Countdown(size=episodes_before_checkpoint)
        self.countdown_till_decay      = Countdown(size=timesteps_before_standard_deviation_decay)
        
        # setup brain
        self.brain = PpoBrain(
            state_dim=product(self.observation_space.shape or [self.observation_space.n]),
            action_dim=product(self.action_space.shape or [self.action_space.n]),
            is_continuous_action_space=(not isinstance(self.action_space, gym.spaces.Discrete)),
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            discount_factor=self.discount_factor,
            number_of_epochs_to_optimize=self.number_of_epochs_to_optimize,
            loss_clamp_boundary=self.loss_clamp_boundary,
            action_std_init=self.action_std_init,
        )
        
        # setup record keeper
        self.records = []
        self.record_keeper = record_keeper if (record_keeper is not None) else RecordKeeper(
            parent_record_keeper=None,
            local_data={"model": "ppo"},
            collection=None,
            records=self.records,
            file_path=self.save_folder,
        )
        attributes_to_record = [
            # agent
            "timesteps_before_weight_update",
            "timesteps_before_standard_deviation_decay",
            # model
            "actor_learning_rate",
            "critic_learning_rate",
            "discount_factor",
            "number_of_epochs_to_optimize",
            "loss_clamp_boundary",
            "action_std_init",
            # world
            "action_space",
            "observation_space",
            "is_continuous_action_space",
        ]
        self.record_keeper = self.record_keeper.sub_record_keeper(**{
            each_attribute: getattr(self, each_attribute)
                for each_attribute in attributes_to_record
        })
    
    @ConnectBody.when_mission_starts
    def when_mission_starts(self):
        self.record_keeper_by_update = self.record_keeper.sub_record_keeper(by_update=True)
        self.record_keeper_by_update.pending_record["reward"] = 0
        self.record_keeper_by_update.pending_record["update_index"] = 0
        
    @ConnectBody.when_episode_starts
    def when_episode_starts(self, episode_index):
        self.accumulated_reward = 0
        self.timestep = 0
        # 
        # logging
        # 
        self.record_keeper.pending_record["updated_weights"] = False
        print("{"+f' "episode": {str(episode_index).ljust(5)},', end="")
    
    @ConnectBody.when_timestep_happens
    def when_timestep_happens(self, timestep_index):
        self.timestep = timestep_index
        # 
        # save the reward for the previous action
        # 
        reward = self.body.get_reward()
        self.accumulated_reward += reward
        self.record_keeper_by_update.pending_record["reward"] += reward
        self.brain.buffer.rewards.append(reward)
        self.brain.buffer.is_terminals.append(False)
        
        # 
        # occasionally update the network
        # 
        if self.countdown_till_update():
            print('"update":true,', end="")
            # save the existing record
            update_index = self.record_keeper_by_update.pending_record["update_index"]
            self.record_keeper_by_update.commit_record()
            # setup for the next record
            self.record_keeper_by_update.pending_record["update_index"] = update_index + 1
            self.record_keeper_by_update.pending_record["reward"] = 0
            # perform the update
            self.brain.update()
        
        # 
        # update action standard deviation
        # 
        # if continuous action space; then decay action std of ouput action distribution
        if self.brain.is_continuous_action_space:
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
        # log
        #
        self.record_keeper.pending_record["accumulated_reward"] = self.accumulated_reward
        self.record_keeper.pending_record["episode_index"] = episode_index
        self.record_keeper.commit_record()
        formatted_reward   = f"{self.accumulated_reward:,.2f}".rjust(8)
        formatted_timestep = f"{self.timestep}".rjust(6)
        print(f' "rewardSum": {formatted_reward}, "numberOfTimesteps": {formatted_timestep}'+"}", flush=True)
        
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