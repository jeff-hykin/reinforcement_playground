from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch
from super_map import LazyDict

from tools.basics import product, flatten
from tools.stat_tools import rolling_average

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


class Agent():
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
        self.observation_size = product(observation_space.shape)
        self.number_of_actions = action_space.n
        self.actor = Actor(self.observation_size, self.number_of_actions)
        self.critic = Critic(self.observation_size)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.discount_factor = 0.99
        self.action_choice_distribution = None
        self.prev_observation = None
        self.action_with_gradient_tracking = None
        self.logging = LazyDict()
        self.logging.should_display = config.get("should_display", True)
        self.logging.episode_rewards = []
        self.logging.episode_critic_losses = []
        self.logging.episode_actor_losses  = []
    
    # 
    # Hooks (Special Names)
    # 
    def when_mission_starts(self):
        self.logging.episode_rewards       = []
        self.logging.episode_critic_losses = []
        self.logging.episode_actor_losses  = []
        
    def when_episode_starts(self, episode_index):
        self.logging.accumulated_reward      = 0
        self.logging.accumulated_critic_loss = 0
        self.logging.accumulated_actor_loss  = 0
    
    def when_timestep_starts(self, timestep_index):
        self.action = self.make_decision(self.observation)
        self.prev_observation = self.observation
        
    def when_timestep_ends(self, timestep_index):
        self.logging.accumulated_reward += self.reward
        self.update_weights(self.compute_advantage(
            reward=self.reward,
            observation=self.prev_observation,
            next_observation=self.observation,
            episode_is_over=self.episode_is_over,
        ))
    
    def when_episode_ends(self, episode_index):
        self.logging.episode_rewards.append(self.logging.accumulated_reward)
        self.logging.episode_critic_losses.append(self.logging.accumulated_critic_loss)
        self.logging.episode_actor_losses.append(self.logging.accumulated_actor_loss)
    
    def when_mission_ends(self,):
        if self.logging.should_display:
            # graph reward results
            ss.DisplayCard("quickLine", rolling_average(self.logging.episode_critic_losses, 5))
            ss.DisplayCard("quickMarkdown", "#### Critic Losses Per Episode")
            ss.DisplayCard("quickLine", rolling_average(self.logging.episode_actor_losses, 5))
            ss.DisplayCard("quickMarkdown", "#### Actor Losses Per Episode")
            ss.DisplayCard("quickLine", rolling_average(self.logging.episode_rewards, 5))
            ss.DisplayCard("quickMarkdown", "#### Rewards Per Episode")
    
    # 
    # Misc Helpers
    # 
    def make_decision(self, observation):
        probs = self.actor(torch.from_numpy(observation).float())
        self.action_choice_distribution = torch.distributions.Categorical(probs=probs)
        self.action_with_gradient_tracking = self.action_choice_distribution.sample()
        return self.action_with_gradient_tracking.item()
    
    def approximate_value_of(self, observation):
        return self.critic(torch.from_numpy(observation).float())
    
    def compute_advantage(self, *, reward, observation, next_observation, episode_is_over):
        return reward + \
            self.approximate_value_of(next_observation)*self.discount_factor*(1-int(episode_is_over))\
            - self.approximate_value_of(observation)
    
    def update_weights(self, advantage):
        critic_loss = advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()
        self.logging.accumulated_critic_loss += critic_loss.item()
        
        actor_loss = -self.action_choice_distribution.log_prob(self.action_with_gradient_tracking)*advantage.detach()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()
        self.logging.accumulated_actor_loss += actor_loss.item()

