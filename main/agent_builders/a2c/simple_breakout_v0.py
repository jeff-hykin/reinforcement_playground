from time import time
import math
from collections import defaultdict
import functools

import torch
from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
from super_map import LazyDict
from gym.wrappers import AtariPreprocessing
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from agent_builders.a2c.baselines_optimizer import RMSpropTFLike
from agent_builders.a2c.frame_que import FrameQue

import tools.stat_tools as stat_tools
from tools.basics import product, flatten, to_pure
from tools.debug import debug
from tools.pytorch_tools import Network, layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential

class ImageNetwork(nn.Module):
    @init.hardware
    def __init__(self, *, input_shape, output_size, **config):
        super().__init__()
        self.dropout_rate    = config.get("dropout_rate", 0.2) # note: not currently in use
        # convert from cv_image shape to torch tensor shape
        color_channels, *_ = self.input_shape = input_shape
        self.layers = Sequential()
        self.layers.add_module('conv1', nn.Conv2d(color_channels, 32, kernel_size=8, stride=4, padding=0))
        self.layers.add_module('conv1_activation', nn.ReLU(inplace=False))
        self.layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0))
        self.layers.add_module('conv2_activation', nn.ReLU(inplace=False))
        self.layers.add_module('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))
        self.layers.add_module('conv3_activation', nn.ReLU(inplace=False))
        self.layers.add_module('flatten', nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.layers.add_module('linear1', nn.Linear(self.size_of_last_layer, 64)) 
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self.layers) == 0 else layer_output_shapes(self.layers, self.input_shape)[-1])
    
    @forward.to_device
    @forward.to_batched_tensor(number_of_dimensions=4)
    def forward(self, X):
        return self.layers.forward(X)

class Actor(nn.Module):
    @init.hardware
    def __init__(self, *, input_size, output_size, **config):
        super().__init__()
        self.layers = Sequential()
        self.layers.add_module('linear1_activation', nn.Tanh()) 
        self.layers.add_module('linear2', nn.Linear(input_size, 32)) 
        self.layers.add_module('linear2_activation', nn.Tanh()) 
        self.layers.add_module('linear3', nn.Linear(32, output_size)) 
        self.layers.add_module('softmax', nn.Softmax(dim=0))
        
    @forward.to_device
    def forward(self, X):
        return self.layers(X)

class Critic(nn.Module):
    @init.hardware
    def __init__(self, *, input_size, **config):
        super().__init__()
        self.layers = Sequential()
        self.layers.add_module('linear1_activation', nn.ReLU()) 
        self.layers.add_module('linear2', nn.Linear(input_size, 32)) 
        self.layers.add_module('linear2_activation', nn.ReLU()) 
        self.layers.add_module('linear3', nn.Linear(32, 1)) 
    
    @forward.to_device
    def forward(self, X):
        return self.layers(X)

class Agent():
    @init.hardware
    def __init__(self, observation_space, action_space, **config):
        # 
        # special
        # 
        self.observation = None     # external world will change this
        self.reward = None          # external world will change this
        self.action = None          # extrenal world will read this
        self.episode_is_over = None # extrenal world will change this
        
        # 
        # regular/misc attributes
        # 
        self.discount_factor      = config.get("discount_factor", 0.99)
        self.actor_learning_rate  = config.get("actor_learning_rate", 0.001)
        self.critic_learning_rate = config.get("critic_learning_rate", 0.001)
        self.dropout_rate         = config.get("dropout_rate", 0.2)
        self.observation_space    = observation_space
        self.number_of_actions = action_space.n
        self.connection_size = 64 # neurons
        self.image_model = ImageNetwork(input_shape=observation_space.shape, output_size =self.connection_size, dropout_rate=self.dropout_rate)
        self.actor       = Sequential(self.image_model, Actor(input_size=self.connection_size , output_size=self.number_of_actions, dropout_rate=self.dropout_rate))
        self.critic      = Sequential(self.image_model, Critic(input_size=self.connection_size, dropout_rate=self.dropout_rate))
        self.adam_actor  = torch.optim.Adam(self.actor.parameters() , lr=self.actor_learning_rate )
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.rms_actor  = RMSpropTFLike(self.actor.parameters() , lr=self.actor_learning_rate, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0, centered=False,) # 1e-5 was a tuned parameter from stable baselines for a2c on atari
        self.rms_critic = RMSpropTFLike(self.critic.parameters(), lr=self.critic_learning_rate, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0, centered=False,) # 1e-5 was a tuned parameter from stable baselines for a2c on atari
        self.action_choice_distribution = None
        self.prev_observation = None
        self.action_with_gradient_tracking = None
        self.buffer = LazyDict()
        self.buffer.observations = []
        self.buffer.rewards = []
        self.buffer.action_log_probabilies = []
        
        self.logging = LazyDict()
        self.logging.should_display = config.get("should_display", False)
        self.logging.live_updates   = config.get("live_updates"  , False)
        self.logging.episode_rewards = []
        self.logging.episode_critic_losses = []
        self.logging.episode_actor_losses  = []
        self.logging.episode_reward_card = None
        self.logging.episode_critic_loss_card = None
        self.logging.episode_actor_loss_card = None
        
        # 
        # class globals
        # 
        if not hasattr(Agent, "agent_number_count"):
            Agent.agent_number_count = 0
        Agent.agent_number_count += 1
        self.agent_number = Agent.agent_number_count
        
        if not hasattr(Agent, "total_number_of_episodes"):
            Agent.total_number_of_episodes = 0
        if not hasattr(Agent, "total_number_of_timesteps"):
            Agent.total_number_of_timesteps = 0
        if not hasattr(Agent, "start_time"):
            Agent.start_time = time()
    
    # 
    # Hooks (Special Names)
    # 
    def when_mission_starts(self):
        self.logging.episode_rewards.clear()
        self.logging.episode_critic_losses.clear()
        self.logging.episode_actor_losses.clear()
        if self.logging.live_updates:
            self.logging.episode_actor_loss_card = ss.DisplayCard("quickLine",[])
            ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Actor Loss, ➡️ Per Episode")
            self.logging.episode_critic_loss_card = ss.DisplayCard("quickLine",[])
            ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Critic Loss, ➡️ Per Episode")
            self.logging.episode_reward_card = ss.DisplayCard("quickLine",[])
            ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Rewards, ➡️ Per Episode")
        
    def when_episode_starts(self, episode_index):
        self.buffer.observations = []
        self.buffer.rewards = []
        self.buffer.action_log_probabilies = []
        self.buffer.observations.append(self.observation) # first observation is ready/valid at the start (if mission is setup correctly)
        
        self.logging.accumulated_reward      = 0
        self.logging.accumulated_critic_loss = 0
        self.logging.accumulated_actor_loss  = 0
    
    def when_timestep_starts(self, timestep_index):
        Agent.total_number_of_timesteps += 1
        self.action = self.make_decision(self.observation)
        self.prev_observation = self.observation
        
    def when_timestep_ends(self, timestep_index):
        reward = self.reward
        # build up value for a large update step later
        self.buffer.observations.append(self.observation)
        self.buffer.rewards.append(reward)
        self.buffer.action_log_probabilies.append(self.action_choice_distribution.log_prob(self.action_with_gradient_tracking))
        # logging
        self.logging.accumulated_reward += reward
    
    def when_episode_ends(self, episode_index):
        self.update_weights_consume_buffer()
        # logging
        Agent.total_number_of_episodes += 1
        self.logging.episode_rewards.append(self.logging.accumulated_reward)
        self.logging.episode_critic_losses.append(self.logging.accumulated_critic_loss)
        self.logging.episode_actor_losses.append(self.logging.accumulated_actor_loss)
        if self.logging.live_updates:
            self.logging.episode_reward_card.send     ([episode_index, self.logging.accumulated_reward      ])
            self.logging.episode_critic_loss_card.send([episode_index, self.logging.accumulated_critic_loss ])
            self.logging.episode_actor_loss_card.send ([episode_index, self.logging.accumulated_actor_loss  ])
        print(f'episode_index = {episode_index}\n    total_number_of_timesteps:{Agent.total_number_of_timesteps}\n    accumulated_reward       :{self.logging.accumulated_reward      }\n    accumulated_critic_loss  :{self.logging.accumulated_critic_loss }\n    accumulated_actor_loss   :{self.logging.accumulated_actor_loss  }', )
    
    def when_mission_ends(self,):
        if self.logging.should_display:
            # graph reward results
            ss.DisplayCard("quickLine", stat_tools.rolling_average(self.logging.episode_actor_losses, 5))
            ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Actor Losses Per Episode")
            ss.DisplayCard("quickLine", stat_tools.rolling_average(self.logging.episode_critic_losses, 5))
            ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Critic Losses Per Episode")
            ss.DisplayCard("quickLine", stat_tools.rolling_average(self.logging.episode_rewards, 5))
            ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Rewards Per Episode")
    
    # 
    # Misc Helpers
    # 
    def make_decision(self, observation):
        probs = self.actor.forward(to_tensor(observation))
        self.action_choice_distribution = torch.distributions.Categorical(probs=probs)
        self.action_with_gradient_tracking = self.action_choice_distribution.sample()
        return to_pure(self.action_with_gradient_tracking)
    
    def approximate_value_of(self, observation):
        return to_pure(self.critic(torch.from_numpy(observation).float()))
    
    # TD0-like
    def _observation_values_vectorized_method(self, value_approximations, rewards_tensor):
        current_approximates = value_approximations[:-1]
        next_approximates = value_approximations[1:]
        # vectorized: (vec + (scalar * vec))
        observation_values = (rewards_tensor + (self.discount_factor*next_approximates)) 
        return observation_values
    
    # TD1/MonteCarlo-like
    def _observation_values_backwards_chain_method(self):
        rewards_tensor = to_tensor(self.buffer.rewards)
        observation_values = torch.zeros_like(rewards_tensor)
        # the last one is just equal to the reward
        observation_values[-1] = rewards_tensor[-1]
        iterable = zip(
            range(len(observation_values)-1),
            rewards_tensor[:-1],
            self.buffer.observations[:-1],
        )
        for reversed_index, each_reward, each_observation in reversed(tuple(iterable)):
            next_estimate = observation_values[reversed_index+1]
            observation_values[reversed_index] = each_reward + self.discount_factor * next_estimate
        return observation_values
    
    def _observation_values_baselines(self):
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
        
    
    def update_weights_consume_buffer(self):
        # send data to device
        value_approximations   = self.critic.forward(to_tensor(self.buffer.observations).to(self.hardware)).squeeze()
        rewards                = to_tensor(self.buffer.rewards).to(self.hardware)
        action_log_probabilies = to_tensor(self.buffer.action_log_probabilies).to(self.hardware)
        
        # 
        # Observation values (not vectorizable)
        # 
        observation_values = self._observation_values_backwards_chain_method().to(self.hardware)
        
        # 
        # compute advantages (self.rewards, self.discount_factor, self.observations)
        # 
        current_approximates = value_approximations[:-1]
        # vectorized: (vec - vec)
        advantages = observation_values - current_approximates
        # last value doesn't have a "next" so manually calculate that
        advantages[-1] = rewards[-1] - value_approximations[-1]
        
        # 
        # loss functions (advantages, self.action_log_probabilies)
        # 
        actor_losses  = -action_log_probabilies * advantages.detach()
        critic_losses = advantages.pow(2) # baselines does F.mse_loss((self.advantages + self.values), self.values) instead for some reason
        actor_episode_loss = actor_losses.mean()
        critic_episode_loss = critic_losses.mean()
        
        self.value_function_coefficient = 0.25 # from stable baselines atari hyperparams TODO: code this properly
        combined_loss = actor_episode_loss + self.value_function_coefficient*critic_episode_loss

        # 
        # update weights accordingly
        # 
        self.rms_actor.zero_grad()
        self.rms_critic.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 0.5 is a default parameter from stable baselines
        self.rms_actor.step()
        self.rms_critic.step()
        self.logging.accumulated_actor_loss += actor_episode_loss.item()
        self.logging.accumulated_critic_loss += critic_episode_loss.item()
        
        # 
        # clear buffer
        # 
        self.buffer = LazyDict()

def default_mission(
        env_name="BreakoutNoFrameskip-v4",
        number_of_episodes=500,
        grayscale=True,
        frame_history=4, # open ai defaults to 4 (VecFrameStack)
        frame_skip=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        screen_size=84,
        discount_factor=0.99,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
    ):
    
    env = AtariPreprocessing(
        gym.make(env_name),
        grayscale_obs=grayscale,
        frame_skip=frame_skip, #
        noop_max=1, # no idea what this is, my best guess is; it is related to a do-dothing action and how many timesteps it does nothing for
        grayscale_newaxis=False, # keeps number of dimensions in observation the same for both grayscale and color (both have 4, b/c of the batch dimension)
    )
    
    mr_bond = Agent(
        # observation_space is little bit hacky
        observation_space=LazyDict(shape=(frame_history, *env.observation_space.shape)),
        action_space=env.action_space,
        # live_updates=True,
        discount_factor=discount_factor,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
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
                improvement = math.pow(absolute_improvement, 1/spike_suppression_magnitude)
            else:
                # just mirror the negative values
                improvement = -math.pow(-absolute_improvement, 1/spike_suppression_magnitude)
            
            improvement_at_this_bucket_level += improvement
        average_improvement = improvement_at_this_bucket_level/(len(bucket_averages)-1) # minus one because its pairwise
        improvements_at_each_bucket_level.append(average_improvement)
    # all split levels given equal weight
    return stat_tools.average(improvements_at_each_bucket_level)

def tune_hyperparams(number_of_episodes_per_trial=100000, fitness_func=fitness_measurement_trend_up):
    import optuna
    # connect the trial-object to hyperparams and setup a measurement of fitness
    objective_func = lambda trial: fitness_func(
        default_mission(
            number_of_episodes=number_of_episodes_per_trial,
            discount_factor=trial.suggest_loguniform('discount_factor', 0.97, 1),
            actor_learning_rate=trial.suggest_loguniform('actor_learning_rate', 0.009, 0.01),
            critic_learning_rate=trial.suggest_loguniform('critic_learning_rate', 0.009, 0.01),    
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
    study = tune_hyperparams()