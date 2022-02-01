from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch
from super_map import LazyDict
import math
from collections import defaultdict
import functools
from gym.wrappers import AtariPreprocessing

from prefabs.baselines_optimizer import RMSpropTFLike

import tools.stat_tools as stat_tools
from tools.basics import product, flatten
from tools.stat_tools import rolling_average
from tools.basics import product, flatten
from tools.debug import debug
from tools.agent_skeleton import Skeleton
from tools.pytorch_tools import Network, layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=0),
        )
    
    def forward(self, X):
        return self.model(X)
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)


class Agent(Skeleton):
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
        self.path                 = config.get("path", None)
        
        self.observation_size = product(observation_space.shape)
        self.number_of_actions = action_space.n
        self.actor = Actor(self.observation_size, self.number_of_actions)
        self.critic = Critic(self.observation_size)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.rms_actor  = RMSpropTFLike(self.actor.parameters() , lr=1e-2, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0, centered=False,) # 1e-5 was a tuned parameter from stable baselines for a2c on atari
        self.rms_critic = RMSpropTFLike(self.critic.parameters(), lr=1e-2, alpha=0.99, eps=1e-5, weight_decay=0, momentum=0, centered=False,) # 1e-5 was a tuned parameter from stable baselines for a2c on atari
        self.action_choice_distribution = None
        self.prev_observation = None
        self.action_with_gradient_tracking = None
        self.buffer = LazyDict()
        self.buffer.observations = []
        self.buffer.rewards = []
        self.buffer.action_log_probabilies = []
        
        self.save, self.load = Network.setup_save_and_load(
            self,
            normal_attributes=["discount_factor", "actor_learning_rate", "critic_learning_rate", "dropout_rate"],
            network_attributes=["actor","critic"],
        ) 
        
        self.logging = LazyDict()
        self.logging.should_display = config.get("should_display", True)
        self.logging.live_updates   = config.get("live_updates"  , False)
        self.logging.episode_rewards = []
        self.logging.episode_critic_losses = []
        self.logging.episode_actor_losses  = []
        self.logging.episode_reward_card = None
        self.logging.episode_critic_loss_card = None
        self.logging.episode_actor_loss_card = None
        
        if not hasattr(Agent, "agent_number_count"):
            Agent.agent_number_count = 0
        Agent.agent_number_count += 1
        self.agent_number = Agent.agent_number_count
    
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
        self.action = self.make_decision(self.observation)
        self.prev_observation = self.observation
        
    def when_timestep_ends(self, timestep_index):
        # build up value for a large update step later
        self.buffer.observations.append(self.observation)
        self.buffer.rewards.append(self.reward)
        self.buffer.action_log_probabilies.append(self.action_choice_distribution.log_prob(self.action_with_gradient_tracking))
        # logging
        self.logging.accumulated_reward += self.reward
    
    def when_episode_ends(self, episode_index):
        self.update_weights_consume_buffer()
        
        # logging
        self.logging.episode_rewards.append(self.logging.accumulated_reward)
        self.logging.episode_critic_losses.append(self.logging.accumulated_critic_loss)
        self.logging.episode_actor_losses.append(self.logging.accumulated_actor_loss)
        if self.logging.live_updates:
            self.logging.episode_reward_card.send     ([episode_index, self.logging.accumulated_reward      ])
            self.logging.episode_critic_loss_card.send([episode_index, self.logging.accumulated_critic_loss ])
            self.logging.episode_actor_loss_card.send ([episode_index, self.logging.accumulated_actor_loss  ])
        print('episode_index = ', episode_index)
        print(f'self.logging.accumulated_reward      :{self.logging.accumulated_reward      }',)
        print(f'self.logging.accumulated_critic_loss :{self.logging.accumulated_critic_loss }',)
        print(f'self.logging.accumulated_actor_loss  :{self.logging.accumulated_actor_loss  }',)
    
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
        probs = self.actor.forward(torch.from_numpy(observation).float())
        self.action_choice_distribution = torch.distributions.Categorical(probs=probs)
        self.action_with_gradient_tracking = self.action_choice_distribution.sample()
        return self.action_with_gradient_tracking.item()
    
    def approximate_value_of(self, observation):
        return self.critic(torch.from_numpy(observation).float()).item()
    
    def _observation_values_vectorized_method(self, value_approximations):
        rewards = to_tensor(self.buffer.rewards)
        
        current_approximates = value_approximations[:-1]
        next_approximates = value_approximations[1:]
        # vectorized: (vec + (scalar * vec))
        observation_values = (rewards + (self.discount_factor*next_approximates)) 
        return observation_values
    
    def _observation_values_backwards_chain_method(self):
        observation_values = torch.zeros(len(self.buffer.rewards))
        # the last one is just equal to the reward
        observation_values[-1] = self.buffer.rewards[-1]
        iterable = zip(
            range(len(observation_values)-1),
            self.buffer.rewards[:-1],
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
        pass
        
    
    def update_weights_consume_buffer(self):
        value_approximations   = self.critic(to_tensor(self.buffer.observations)).squeeze()
        
        # 
        # Observation values (not vectorizable)
        # 
        observation_values = self._observation_values_backwards_chain_method()
        
        # 
        # compute advantages (self.rewards, self.discount_factor, self.observations)
        # 
        current_approximates = value_approximations[:-1]
        rewards = to_tensor(self.buffer.rewards)
        # vectorized: (vec - vec)
        advantages = observation_values - current_approximates
        # last value doesn't have a "next" so manually calculate that
        advantages[-1] = rewards[-1] - value_approximations[-1]
        
        # 
        # loss functions (advantages, self.action_log_probabilies)
        # 
        action_log_probabilies = to_tensor(self.buffer.action_log_probabilies)
        actor_losses  = -action_log_probabilies * advantages.detach()
        critic_losses = advantages.pow(2) # baselines does F.mse_loss((self.advantages + self.values), self.values) instead for some reason
        actor_episode_loss = actor_losses.mean()
        critic_episode_loss = critic_losses.mean()
        
        # 
        # update weights accordingly
        # 
        self.adam_actor.zero_grad()
        self.adam_critic.zero_grad()
        actor_episode_loss.backward()
        critic_episode_loss.backward()
        self.adam_actor.step()
        self.adam_critic.step()
        self.logging.accumulated_actor_loss += actor_episode_loss.item()
        self.logging.accumulated_critic_loss += critic_episode_loss.item()
        
        # 
        # clear buffer
        # 
        self.buffer = LazyDict()


