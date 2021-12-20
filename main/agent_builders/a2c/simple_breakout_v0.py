from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch
from super_map import LazyDict

from tools.basics import product, flatten
from tools.stat_tools import rolling_average
from tools.pytorch_tools import Network, layer_output_shapes, opencv_image_to_torch_image

class Actor(nn.Module):
    def __init__(self, *, input_shape, output_size):
        super().__init__()
        color_channels = 3
        # convert from cv_image shape to torch tensor shape
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.layers = nn.Sequential()
        self.layers.add_module('conv1', nn.Conv2d(color_channels, 32, kernel_size=8, stride=4, padding=0))
        self.layers.add_module('conv1_activation', nn.ReLU())
        self.layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0))
        self.layers.add_module('conv2_activation', nn.ReLU())
        self.layers.add_module('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))
        self.layers.add_module('conv3_activation', nn.ReLU())
        self.layers.add_module('flatten', nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.layers.add_module('linear1', nn.Linear(self.size_of_last_layer, 64)) 
        self.layers.add_module('linear1_activation', nn.Tanh()) 
        self.layers.add_module('linear2', nn.Linear(64, 32)) 
        self.layers.add_module('linear2_activation', nn.Tanh()) 
        self.layers.add_module('linear3', nn.Linear(32, output_size)) 
        self.layers.add_module('softmax', nn.Softmax(dim=0))
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self.layers) == 0 else layer_output_shapes(self.layers, self.input_shape)[-1])
        
    def forward(self, X):
        X = opencv_image_to_torch_image(X)
        X = X.reshape((-1,*X.shape)) # add a dimension to create a "batch" of 1
        return self.layers(X)
    
class Critic(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        color_channels = 3
        # convert from cv_image shape to torch tensor shape
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.layers = nn.Sequential()
        self.layers.add_module('conv1', nn.Conv2d(color_channels, 32, kernel_size=8, stride=4, padding=0))
        self.layers.add_module('conv1_activation', nn.ReLU())
        self.layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0))
        self.layers.add_module('conv2_activation', nn.ReLU())
        self.layers.add_module('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))
        self.layers.add_module('conv3_activation', nn.ReLU())
        self.layers.add_module('flatten', nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.layers.add_module('linear1', nn.Linear(self.size_of_last_layer, 64)) 
        self.layers.add_module('linear1_activation', nn.ReLU()) 
        self.layers.add_module('linear2', nn.Linear(64, 32)) 
        self.layers.add_module('linear2_activation', nn.ReLU()) 
        self.layers.add_module('linear3', nn.Linear(32, 1)) 
        
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self.layers) == 0 else layer_output_shapes(self.layers, self.input_shape)[-1])
        
    def forward(self, X):
        X = opencv_image_to_torch_image(X)
        X = X.reshape((-1,*X.shape))
        return self.layers(X)


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
        self.number_of_actions = action_space.n
        self.actor = Actor(input_shape=observation_space.shape, output_size=self.number_of_actions)
        self.critic = Critic(input_shape=observation_space.shape)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.discount_factor = 0.99
        self.action_choice_distribution = None
        self.prev_observation = None
        self.action_with_gradient_tracking = None
        self.logging = LazyDict()
        self.logging.should_display = config.get("should_display", True)
        self.logging.live_updates   = config.get("live_updates"  , False)
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
        if self.logging.live_updates:
            self.logging.card = ss.DisplayCard("quickLine",[])
            ss.DisplayCard("quickMarkdown", "#### Live Rewards Per Episode")
        
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
        self.logging.card.send([episode_index, self.logging.accumulated_reward])
    
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

def fitness_measurement(episode_rewards, spike_suppression_magnitude=8, granuality_branching_factor=4, min_bucket_size=6, max_bucket_proportion=0.5):
    # measure: should trend up, more improvement is better, but trend is most important
    # trend is measured at recusively granular levels: default splits of (1/4th's, 1/16th's, 1/64th's ...)
    # the default max proportion (0.5) prevents bucket from being more than 50% of the full list (set to max to 1 to allow entire list as first "bucket")
    recursive_splits_list = stat_tools.recursive_splits(
        episode_rewards,
        branching_factor=granuality_branching_factor,
        min_size=min_bucket_size,
        max_proportion=max_bucket_proportion, 
    )
    improvements_at_each_bucket_level = []
    for buckets in recursive_splits_list:
        bucket_averages = [ stat_tools.average(each_bucket) for each_bucket in buckets ]
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

            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            mr_bond.when_timestep_ends(timestep_index)
                
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    env.close()