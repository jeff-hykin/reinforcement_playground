from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch
from super_map import LazyDict
import math
from collections import defaultdict
import functools

import tools.stat_tools as stat_tools
from tools.basics import product, flatten
from tools.debug import debug
from tools.pytorch_tools import Network, layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential

class ImageNetwork(nn.Module):
    @init.hardware
    def __init__(self, *, input_shape, output_size, **config):
        super().__init__()
        self.dropout_rate    = config.get("dropout_rate", 0.2) # note: not currently in use
        
        color_channels = 3
        # convert from cv_image shape to torch tensor shape
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
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
    @forward.to_batched_tensor(4)
    @forward.from_opencv_image_to_torch_image
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
    
agent_number = 1
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
        self.discount_factor      = config.get("discount_factor", 0.99)
        self.actor_learning_rate  = config.get("actor_learning_rate", 0.001)
        self.critic_learning_rate = config.get("critic_learning_rate", 0.001)
        self.dropout_rate         = config.get("dropout_rate", 0.2)
        self.number_of_actions = action_space.n
        self.connection_size = 64 # neurons
        self.image_model = ImageNetwork(input_shape = observation_space.shape, output_size  = self.connection_size  , dropout_rate = self.dropout_rate)
        self.actor       = Sequential(self.image_model, Actor(input_size=self.connection_size , output_size=self.number_of_actions, dropout_rate=self.dropout_rate))
        self.critic      = Sequential(self.image_model, Critic(input_size=self.connection_size, dropout_rate=self.dropout_rate))
        self.adam_actor  = torch.optim.Adam(self.actor.parameters() , lr=self.actor_learning_rate )
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.action_choice_distribution = None
        self.prev_observation = None
        self.action_with_gradient_tracking = None
        self.episode = LazyDict()
        self.episode.observations = []
        self.episode.rewards = []
        self.logging = LazyDict()
        self.logging.should_display = config.get("should_display", True)
        self.logging.live_updates   = config.get("live_updates"  , False)
        self.logging.episode_rewards = []
        self.logging.episode_critic_losses = []
        self.logging.episode_actor_losses  = []
        self.logging.episode_reward_card = None
        self.logging.episode_critic_loss_card = None
        self.logging.episode_actor_loss_card = None
        
        global agent_number
        self.agent_number = agent_number
        agent_number += 1
    
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
        self.episode.observations.clear()
        self.episode.rewards.clear()
        self.episode.observations.append(self.observation) # first observation is ready/valid at the start (if mission is setup correctly)
        
        self.logging.accumulated_reward      = 0
        self.logging.accumulated_critic_loss = 0
        self.logging.accumulated_actor_loss  = 0
    
    def when_timestep_starts(self, timestep_index):
        self.action = self.make_decision(self.observation)
        self.prev_observation = self.observation
        
    def when_timestep_ends(self, timestep_index):
        self.episode.observations.append(self.observation)
        self.episode.rewards.append(self.reward)
        # self.trajectory.append((timestep_index, self.prev_observation, self.action_choice_distribution, self.reward, self.observation, self.episode_is_over))
        self.update_weights(self.compute_advantage(
            reward=self.reward,
            observation=self.prev_observation,
            next_observation=self.observation,
            episode_is_over=self.episode_is_over,
        ))
        # logging
        self.logging.accumulated_reward += self.reward
    
    def when_episode_ends(self, episode_index):
        # one giant batch of observations
        
        # value_approximations = self.critic(to_tensor(self.episode.observations).to(self.device))
        # rewards              = to_tensor(self.episode_rewards)
        
        # # real reward + discounted estimated-reward (multiplied by 0 if last timestep)
        # observation_values = [     reward + (self.discount_factor*approximated_value)     for reward, approximated_value in zip(rewards, value_approximations) ]
        # # sometimes the last value is set to zero (env_stopped_the_episode is false when max_number_of_timesteps is hit)
        # observation_values[-1] = 0 if env_stopped_the_episode else observation_values[-1]
        # # difference between the partially-observed value and the pure-estimate value
        # deltas = np.array([    each_observation_value - value_approximator(each_observation)    for each_observation_value, each_observation zip(observation_values, observations) ])
        
        # advantage = reward + ((1.0 - done) * discount_factor * value_approximator(next_state)) - value_approximator(state)
        # # one-hot encoding for actions (matrix)
        # actions_one_hot = np.zeros([len(actions), self.env.action_space.n])
        # actions_one_hot[np.arange(len(actions)), actions] = 1
        # FIXME: self.update_weights
        
        # logging
        self.logging.episode_rewards.append(self.logging.accumulated_reward)
        self.logging.episode_critic_losses.append(self.logging.accumulated_critic_loss)
        self.logging.episode_actor_losses.append(self.logging.accumulated_actor_loss)
        if self.logging.live_updates:
            self.logging.episode_reward_card.send     ([episode_index, self.logging.accumulated_reward      ])
            self.logging.episode_critic_loss_card.send([episode_index, self.logging.accumulated_critic_loss ])
            self.logging.episode_actor_loss_card.send ([episode_index, self.logging.accumulated_actor_loss  ])
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
        return self.critic(torch.from_numpy(observation).float())
    
    def compute_advantage(self, *, reward, observation, next_observation, episode_is_over):
        return reward + \
            self.approximate_value_of(next_observation)*self.discount_factor*(1-int(episode_is_over))\
            - self.approximate_value_of(observation)
    
    def update_weights(self, advantage):
        actor_loss = -self.action_choice_distribution.log_prob(self.action_with_gradient_tracking)*advantage.clone().detach()
        critic_loss = advantage.pow(2).mean()
        self.adam_actor.zero_grad()
        self.adam_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.adam_actor.step()
        self.adam_critic.step()
        self.logging.accumulated_actor_loss += actor_loss.item()
        self.logging.accumulated_critic_loss += critic_loss.item()
        

def default_mission(env_name="BreakoutNoFrameskip-v4", number_of_episodes=500, discount_factor=0.99, actor_learning_rate=0.001, critic_learning_rate=0.001):
    env = gym.make(env_name)
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        live_updates=True,
        discount_factor=discount_factor,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
    )
    mr_bond.when_mission_starts()
    for episode_index in range(number_of_episodes):
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
    mr_bond.when_mission_ends()
    env.close()
    return mr_bond

def fitness_measurement_average_reward(episode_rewards):
    return stat_tools.average(episode_rewards)

def fitness_measurement_trend_up(episode_rewards, spike_suppression_magnitude=8, granuality_branching_factor=3, min_bucket_size=6, max_bucket_proportion=0.5):
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

def tune_hyperparams(initial_number_of_episodes_per_trial=10, episode_compounding_rate=1.07, fitness_func=fitness_measurement_average_reward):
    import optuna
    # setup the number of episodes
    def increasing_number_of_episodes():
        number_of_episodes = initial_number_of_episodes_per_trial
        while True:
            yield int(number_of_episodes)
            # increase in size as more trials are done
            number_of_episodes *= episode_compounding_rate
    incrementally_more_episodes = increasing_number_of_episodes()
        
    # connect the trial-object to hyperparams and setup a measurement of fitness
    objective_func = lambda trial: fitness_func(
        default_mission(
            number_of_episodes=next(incrementally_more_episodes),
            discount_factor=trial.suggest_loguniform('discount_factor', 0.8, 1),
            actor_learning_rate=trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-1),
            critic_learning_rate=trial.suggest_loguniform('critic_learning_rate', 1e-5, 1e-1),    
        ).logging.episode_rewards
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=100)
    return study

# 
# do mission if run directly
# 
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    study = tune_hyperparams()