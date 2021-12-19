from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch

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
            nn.Softmax()
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
    def __init__(self, observation_space, action_space):
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
        self.episode_rewards = []
        self.action_choice_distribution = None
        self.prev_observation = None
        self.action_with_gradient_tracking = None
    
    # 
    # Hooks (Special Names)
    # 
    def when_mission_starts(self):
        self.episode_rewards = []
        
    def when_episode_starts(self, episode_index):
        self.accumulated_reward = 0
    
    def when_timestep_starts(self, timestep_index):
        self.action = self.make_decision(self.observation)
        self.prev_observation = self.observation
        
    def when_timestep_ends(self, timestep_index):
        self.accumulated_reward += self.reward
        advantage = self.compute_advantage(
            reward=self.reward,
            observation=self.prev_observation,
            next_observation=self.observation,
            episode_is_over=self.episode_is_over,
        )
        self.update_weights(advantage)
    
    def when_episode_ends(self, episode_index):
        # record the rewards
        self.episode_rewards.append(self.accumulated_reward)
    
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
        
        actor_loss = -self.action_choice_distribution.log_prob(self.action_with_gradient_tracking)*advantage.detach()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()

# 
# setup mission
# 
env = gym.make("CartPole-v1")
mr_bond = Agent(
    observation_space=env.observation_space,
    action_space=env.action_space
)
mr_bond.when_mission_starts(episode_index)
for episode_index in range(500):
    mr_bond.episode_is_over = False
    mr_bond.observation = env.reset()
    mr_bond.when_episode_starts(episode_index)
    
    timestep_index = -1
    while not mr_bond.episode_is_over:
        timestep_index += 1
        
        mr_bond.when_timestep_starts(timestep_index)
        mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
        mr_bond.when_timestep_ends(timestep_index)
            
    mr_bond.when_episode_ends(episode_index)
mr_bond.when_mission_ends(episode_index)
env.close()

ss.DisplayCard("quickLine", rolling_average(mr_bond.episode_rewards, 5))