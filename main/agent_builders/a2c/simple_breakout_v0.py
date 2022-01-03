from time import time
import math
from collections import defaultdict
import functools

import torch
from torch import nn
import gym
import numpy as np
# import silver_spectacle as ss
from super_map import LazyDict
import math
from collections import defaultdict
import functools
from stable_baselines3.common.vec_env import VecFrameStack
from gym.wrappers import AtariPreprocessing
from stable_baselines3.common.env_util import make_atari_env

from agent_builders.a2c.baselines_optimizer import RMSpropTFLike
from agent_builders.a2c.frame_que import FrameQue
from agent_builders.a2c.schedulers import LearningRateScheduler

import tools.stat_tools as stat_tools
from tools.basics import product, flatten, to_pure
from tools.debug import debug
from tools.pytorch_tools import layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential

class Agent():
    @init.hardware
    def __init__(self, observation_space, action_space, **config):
        # 
        # special
        # 
        self.observation = None     # external world will change this
        self.reward = None          # external world will change this
        self.action = None          # external world will read this
        self.episode_is_over = None # external world will change this
        
        # 
        # regular/misc attributes
        # 
        self.discount_factor         = config.get("discount_factor", 0.99)
        self.learning_rate           = config.get("learning_rate", 0.0007)
        self.gradient_clip_threshold = config.get("gradient_clip_threshold", 0.5) # 0.5 is for atari in https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
        self.actor_weight            = config.get("actor_weight", 1) # 1 basically means "as-is"
        self.critic_weight           = config.get("critic_weight", 0.25) # 0.25 is for atari in https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
        self.entropy_weight          = config.get("entropy_weight", 0.01) # 0.01 is for atari in https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
        self.dropout_rate            = config.get("dropout_rate", 0.2)
        self.number_of_actions = action_space.n
        self.action_choice_distribution    = None
        self.action_with_gradient_tracking = None
        self.observation_value_estimate    = None
        
        # 
        # model definition
        # 
        self.connection_size = 512 # neurons
        self.model = Agent.Network(input_shape=observation_space.shape, latent_size=self.connection_size, dropout_rate=self.dropout_rate)
        self.rms_optimizer = RMSpropTFLike(self.model.parameters(), lr=0, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0, centered=False,) # 1e-5 was a tuned parameter from stable baselines for a2c on atari
        self.learning_rate_scheduler = LearningRateScheduler(
            value_function=self.learning_rate,
            optimizers=[ self.rms_optimizer ],
        )
        
        # 
        # tools with hooks (which never modify the agent)
        # 
        self.buffer = Agent.Buffer(agent=self)
        self.logger = Agent.Logger(agent=self)
    
    # 
    # sub-classes
    # 
    
    class Network(nn.Module):
        @init.hardware
        def __init__(self, *, input_shape, latent_size, **config):
            super().__init__()
            self.dropout_rate    = config.get("dropout_rate", 0.2) # note: not currently in use
            # get color channels and set input shape
            color_channels, *_ = self.input_shape = input_shape
            self.layers = Sequential()
            self.layers.add_module('conv1', nn.Conv2d(color_channels, 32, kernel_size=8, stride=4, padding=0))
            self.layers.add_module('conv1_activation', nn.ReLU())
            self.layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0))
            self.layers.add_module('conv2_activation', nn.ReLU())
            self.layers.add_module('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))
            self.layers.add_module('conv3_activation', nn.ReLU())
            self.layers.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1)) # 1 => skip the first dimension because thats the batch dimension
            self.layers.add_module('linear1', nn.Linear(in_features=self.size_of_last_layer, out_features=latent_size, bias=True)) 
            
            self.actor  = Sequential(
                nn.Linear(in_features=latent_size, out_features=4, bias=True),
                nn.Softmax(dim=0), # not sure why but stable baselines doesn't have Softmax (removing it causes issues with probability distributions though)
            )
            self.critic = Sequential(
                nn.Linear(in_features=latent_size, out_features=1, bias=True),
                nn.ReLU(),
            )
        
        @property
        def size_of_last_layer(self):
            with torch.no_grad():
                return product(self.input_shape if len(self.layers) == 0 else layer_output_shapes(self.layers, self.input_shape)[-1])
        
        @forward.to_device
        @forward.to_batched_tensor(number_of_dimensions=4) # batch_size, color_channels, image_width, image_height
        def forward(self, images):
            vectors_of_features = self.layers.forward(images)
            critic_evaluations = self.critic.forward(vectors_of_features)
            vectors_of_action_probability_vectors = self.actor.forward(vectors_of_features)
            action_distributions  = torch.distributions.Categorical(probs=vectors_of_action_probability_vectors)
            return action_distributions, critic_evaluations
    
    class Buffer:
        # depends on:
        #     self.agent.reward
        #     self.agent.action_choice_distribution
        #     self.agent.action_with_gradient_tracking
        #     self.agent.observation_value_estimate
        #     self.agent.action_entropy
        def __init__(self, agent):
            self.agent = agent
            self.clear()
        
        def clear(self):
            self.rewards                     = []
            self.action_log_probabilies      = []
            self.observation_value_estimates = []
            self.each_action_entropy         = []
            self.was_last_episode_reward     = []
        
        def when_episode_starts(self, episode_index):
            self.clear()
        
        def when_timestep_starts(self, timestep_index):
            pass
            
        def when_timestep_ends(self, timestep_index):
            self.rewards.append(self.agent.reward)
            self.action_log_probabilies.append(self.agent.action_choice_distribution.log_prob(self.agent.action_with_gradient_tracking))
            self.observation_value_estimates.append(self.agent.observation_value_estimate)
            self.each_action_entropy.append(self.agent.action_entropy)
            self.was_last_episode_reward.append(False)
        
        def when_episode_ends(self, episode_index):
            # correct the buffer value since the episode ended
            self.was_last_episode_reward[-1] = True
        
        def when_weight_update_starts(self):
            pass
        
        def when_weight_update_ends(self):
            self.clear()
    
    class Logger:
        # depends on:
        #      self.agent.reward
        #      self.agent.loss
        def __init__(self, agent, **config):
            self.agent = agent
            
            self.should_display   = config.get("should_display"  , False)
            self.live_updates     = config.get("live_updates"    , False)
            self.smoothing_amount = config.get("smoothing_amount", 5    )
            self.episode_rewards = []
            self.episode_losses  = []
            self.episode_reward_card = None
            self.episode_loss_card = None
            
            # init class attributes if doesn't already have them
            self.static = Agent.Logger.static = LazyDict(
                agent_number_count=0,
                total_number_of_episodes=0,
                total_number_of_timesteps=0,
                start_time=time(),
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
            self.accumulated_reward += self.agent.reward
        
        def when_episode_ends(self, episode_index):
            # logging
            self.episode_rewards.append(self.accumulated_reward)
            self.episode_losses.append(self.accumulated_loss)
            if self.live_updates:
                self.episode_reward_card.send     ([episode_index, self.accumulated_reward      ])
                self.episode_loss_card.send ([episode_index, self.accumulated_loss  ])
            print('episode_index = ', episode_index)
            print(f'    average_episode_time :{(time()-self.static.start_time)/self.static.total_number_of_episodes}',)
            print(f'    accumulated_reward   :{self.accumulated_reward      }',)
            print(f'    accumulated_loss     :{self.accumulated_loss  }',)
        
        def when_mission_ends(self,):
            if self.should_display:
                # graph reward results
                ss.DisplayCard("quickLine", stat_tools.rolling_average(self.episode_losses, self.smoothing_amount))
                ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Losses Per Episode")
                ss.DisplayCard("quickLine", stat_tools.rolling_average(self.episode_rewards, self.smoothing_amount))
                ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Rewards Per Episode")
        
        def when_weight_update_starts(self):
            pass

        def when_weight_update_ends(self):
            self.accumulated_loss += self.agent.loss.item()
    
    # 
    # Hooks (Special Names)
    # 
    def when_mission_starts(self):
        self.logger.when_mission_starts()
        
    def when_episode_starts(self, episode_index):
        self.learning_rate_scheduler.when_episode_starts(episode_index)
        self.buffer.when_episode_starts(episode_index)
        self.logger.when_episode_starts(episode_index)
        
    def when_timestep_starts(self, timestep_index):
        self.learning_rate_scheduler.when_timestep_starts(timestep_index)
        self.logger.when_timestep_starts(timestep_index)
        
        # 
        # run the model
        # 
        action_choice_distributions, critic_evaluations = self.model.forward(
            to_tensor(self.observation)
        )
        self.action_choice_distribution    = action_choice_distributions
        self.observation_value_estimate    = critic_evaluations # just keeping for buffer/loss calculation
        self.action_entropy                = self.action_choice_distribution.entropy() # just keeping for buffer/loss calculation
        self.action_with_gradient_tracking = self.action_choice_distribution.sample()
        
        # 
        # choose an action
        # 
        self.action = to_pure(self.action_with_gradient_tracking)
        
    def when_timestep_ends(self, timestep_index):
        self.buffer.when_timestep_ends(timestep_index)
        self.logger.when_timestep_ends(timestep_index)
    
    def when_episode_ends(self, episode_index):
        self.buffer.when_episode_ends(episode_index)
        
        # 
        # update weights
        # 
        self.update_weights()
        
        self.logger.when_episode_ends(episode_index)
        
    
    def when_mission_ends(self):
        self.logger.when_mission_ends()
    
    # 
    # Misc Helpers
    # 
    def update_weights(self):
        # call start hooks
        self.learning_rate_scheduler.when_weight_update_starts()
        self.buffer.when_weight_update_starts()
        self.logger.when_weight_update_starts()
        
        # convert to tensors
        rewards                     = to_tensor(self.buffer.rewards                    ).to(self.hardware)
        action_log_probabilies      = to_tensor(self.buffer.action_log_probabilies     ).to(self.hardware)
        observation_value_estimates = to_tensor(self.buffer.observation_value_estimates).to(self.hardware)
        each_action_entropy         = to_tensor(self.buffer.each_action_entropy        ).to(self.hardware)
        was_last_episode_reward     = to_tensor(self.buffer.was_last_episode_reward    ).to(self.hardware)
        
        # 
        # Observation values (not vectorizable)
        # 
        improved_observation_values = self._improved_observation_values_backwards_chain_method(
            rewards=rewards,
            discount_factor=self.discount_factor,
        ).to(self.hardware)
        
        # 
        # compute advantages
        # 
        advantages = improved_observation_values - observation_value_estimates
        
        # 
        # loss (needs: advantages, action_log_probabilies, each_action_entropy)
        # 
        actor_losses  = -action_log_probabilies * advantages.detach()
        critic_losses = advantages.pow(2) # baselines does F.mse_loss((self.advantages + self.values), self.values) instead for some reason
        self.actor_episode_loss   = self.actor_weight   * actor_losses.mean()
        self.critic_episode_loss  = self.critic_weight  * critic_losses.mean()
        self.entropy_episode_loss = self.entropy_weight * -torch.mean(each_action_entropy)
        self.loss = self.actor_episode_loss + self.critic_episode_loss + self.entropy_episode_loss

        # 
        # update weights accordingly
        # 
        self.rms_optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_threshold)
        self.rms_optimizer.step()
        
        # call end hooks
        self.buffer.when_weight_update_ends()
        self.logger.when_weight_update_ends()
    
    # TD0-like
    def _improved_observation_values_vectorized_method(self, observation_value_estimates, rewards_tensor):
        current_approximates = observation_value_estimates[:-1]
        next_approximates = observation_value_estimates[1:]
        # vectorized: (vec + (scalar * vec))
        observation_values = (rewards_tensor + (self.discount_factor*next_approximates)) 
        return observation_values
    
    # TD1/MonteCarlo-like
    def _improved_observation_values_backwards_chain_method(self, rewards, discount_factor):
        observation_values = torch.zeros(len(rewards))
        # the last one is just equal to the reward
        observation_values[-1] = rewards[-1]
        iterable = zip(
            range(len(observation_values)-1),
            rewards[:-1],
        )
        for reversed_index, each_reward in reversed(tuple(iterable)):
            next_estimate = observation_values[reversed_index+1]
            observation_values[reversed_index] = each_reward + discount_factor * next_estimate
        return observation_values
    
    def _improved_observation_values_baselines(self):
        # https://github.com/DLR-RM/stable-baselines3/blob/3b68dc731219f112ccc2a6745f216bca701080bb/stable_baselines3/ppo/ppo.py#L198
        # seems like this method is complicated
        # - they store the raw observation on the rollout buffer
        # - then they do feature extraction (which depends/changes with the config)
        # - ^ this is basically preprocessing
        # - they then use an mlp_extractor, which IDK what that is 
        # - its possible this whole thing is just a way to calculate the action distribution: which is something I just store/have directly
        # - advantages are stored in the rollout buffer somehow. Found it: https://github.com/DLR-RM/stable-baselines3/blob/3b68dc731219f112ccc2a6745f216bca701080bb/stable_baselines3/common/buffers.py#L349
        #    - does seem they are calculated in a reversed manner
        #    - has gae_lambda as a tradeoff between TD0-style <-> MonteCarlo style updates
        # - then theres some computation that utilizes entropy 
        # - they also list the learning rate as "linear" which makes me think its a dynamic rate
        pass
        

def default_mission(
        env_name="BreakoutNoFrameskip-v4",
        number_of_episodes=500,
        grayscale=True,
        frame_history=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        screen_size=84,
        discount_factor=0.99,
        learning_rate=0.0007, # open ai defaults to 0.0007 for a2c
    ):
    
    env = AtariPreprocessing(
        gym.make(env_name),
        grayscale_obs=grayscale,
        frame_skip=frame_sample_rate,
        noop_max=1, # no idea what this is, my best guess is; it is related to a do-dothing action and how many timesteps it does nothing for
        grayscale_newaxis=False, # keeps number of dimensions in observation the same for both grayscale and color (both have 4, b/c of the batch dimension)
    )
    
    mr_bond = Agent(
        # observation_space is little bit hacky
        observation_space=LazyDict(shape=(frame_history, *env.observation_space.shape)),
        action_space=env.action_space,
        # live_updates=True,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
    )
    
    mr_bond.when_mission_starts()
    for episode_index in range(number_of_episodes):
        mr_bond.episode_is_over = False
        # observation is a que of frames
        mr_bond.observation = FrameQue(que_size=frame_history, frame_shape=env.observation_space.shape) # fram que should probably be a wrapper around atari
        # add the latest frame
        mr_bond.observation.add(env.reset())
        mr_bond.when_episode_starts(episode_index)
        
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            mr_bond.when_timestep_starts(timestep_index)
            latest_frame, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            # push in the newest frame
            mr_bond.observation.add(to_tensor(latest_frame))
            
            mr_bond.when_timestep_ends(timestep_index)
                
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    env.close()
    return mr_bond

def fitness_measurement_average_reward(episode_rewards):
    return stat_tools.average(episode_rewards)

def fitness_measurement_trend_up(episode_rewards, spike_suppression_magnitude=8, granuality_branching_factor=3, min_bucket_size=5, max_bucket_proportion=0.65):
    # measure: should trend up, more improvement is better, but trend is most important
    # trend is measured at recusively granular levels: default splits of (1/3th's, 1/9th's, 1/27th's ...)
    # the default max proportion (0.5) prevents bucket from being more than 50% of the full list (set to max to 1 to allow entire list as first "bucket")
    recursive_splits_list = stat_tools.recursive_splits(
        episode_rewards,
        branching_factor=granuality_branching_factor,
        min_size=min_bucket_size,
        max_proportion=max_bucket_proportion, 
    )
    improvements_at_each_bucket_level = []
    for buckets in recursive_splits_list:
        bucket_averages = [ stat_tools.average(each_bucket) for each_bucket in buckets if len(each_bucket) > 0 ]
        improvement_at_this_bucket_level = 0
        for prev_average, next_average in stat_tools.pairwise(bucket_averages):
            absolute_improvement = next_average - prev_average
            # pow is being used as an Nth-root
            # and Nth-root is used because we don't care about big spikes
            # we want to measure general improvement, while still keeping the property that more=>better
            if absolute_improvement > 0:
                improvement = matorch.pow(absolute_improvement, 1/spike_suppression_magnitude)
            else:
                # just mirror the negative values
                improvement = -matorch.pow(-absolute_improvement, 1/spike_suppression_magnitude)
            
            improvement_at_this_bucket_level += improvement
        average_improvement = improvement_at_this_bucket_level/(len(bucket_averages)-1) # minus one because its pairwise
        improvements_at_each_bucket_level.append(average_improvement)
    # all split levels given equal weight
    return stat_tools.average(improvements_at_each_bucket_level)

def tune_hyperparams(number_of_episodes_per_trial=100_000, fitness_func=fitness_measurement_trend_up):
    import optuna
    # connect the trial-object to hyperparams and setup a measurement of fitness
    objective_func = lambda trial: fitness_func(
        default_mission(
            number_of_episodes=number_of_episodes_per_trial,
            discount_factor=trial.suggest_loguniform('discount_factor', 0.990, 0.991),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.00070, 0.00071),
        ).logging.episode_rewards
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=50)
    return study

# 
# do mission if run directly
# 
if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True) # comment out unless debugging 
    default_mission(
        number_of_episodes=100_000,
    )